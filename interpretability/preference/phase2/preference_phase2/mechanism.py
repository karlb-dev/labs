"""Mechanism v3: scenario-local relative-choice-value directions fitted
from RANDOMIZED context advantage, never from observed choices
(plan Part VI; addendum F pins).

CPU-testable core: pairing, direction estimation, decoder gates (E11),
stability, neutral relevance, identifiability precheck (§36), site/layer
score (§37). GPU assays (patching, addition, removal, propagation, dose
guards) consume these outputs through mechanism_run.

States are float32 arrays keyed by (item_id, site, depth); every function
takes explicit row lists + a states lookup so synthetic worlds inject
fake geometry through the exact production paths.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .canonical import stable_seed
from .stats import SEED_BASE

N_PERMUTATIONS = 1000
N_BOOT_DIRECTIONS = 100
DECODER_CORR_MIN = 0.40
DECODER_AUC_MIN = 0.70
SPLIT_HALF_COS_MEDIAN_MIN = 0.60
MIN_TRAIN_INC = 16
MIN_VAL_INC = 8
MIN_HOLDOUT_INC = 8

# §37 selection score weights
SCORE_WEIGHTS = {
    "decoder_corr": 0.30, "auc_gap": 0.20, "split_half": 0.20,
    "neutral_corr": 0.20, "cross_codebook": 0.10,
}
# §34 site orderings for the upstream tie-break
SITE_ORDER = ("context_end", "option_a_end", "option_b_end", "menu_end",
              "response_instruction_start", "final_prompt_token")

StateFn = Callable[[str], np.ndarray]  # item_id -> state vector


def _pair_key(r: dict[str, Any]) -> tuple:
    return (r["incidental_id"], r["display_order"], r["code_map_index"],
            r.get("paraphrase_id", 0), abs(int(r["context_strength"])),
            r.get("codebook_pair_id"))


def matched_pairs(rows: list[dict[str, Any]]) -> list[tuple[dict, dict]]:
    """A-favor/B-favor row pairs matched within incidental and surface at
    equal |s| (addendum F: |s| in {1,2} pooled with |s|-matched pairing)."""
    plus: dict[tuple, dict] = {}
    minus: dict[tuple, dict] = {}
    for r in rows:
        s = int(r["context_strength"])
        if s == 0:
            continue
        (plus if s > 0 else minus)[_pair_key(r)] = r
    return [(plus[k], minus[k]) for k in sorted(plus.keys() & minus.keys(),
                                                key=str)]


def fit_direction(rows: list[dict[str, Any]], get_state: StateFn,
                  *, split: str = "train") -> dict[str, Any]:
    """d = unit(E[h_Afavor - h_Bfavor]) over matched pairs in ``split``."""
    use = [r for r in rows if r["incidental_split"] == split]
    pairs = matched_pairs(use)
    if not pairs:
        return {"ok": False, "reason": "no matched pairs"}
    deltas = np.stack([
        get_state(p["item_id"]).astype(np.float64)
        - get_state(m["item_id"]).astype(np.float64)
        for p, m in pairs])
    raw = deltas.mean(axis=0)
    norm = float(np.linalg.norm(raw))
    if not np.isfinite(norm) or norm == 0:
        return {"ok": False, "reason": "zero/non-finite direction"}
    return {"ok": True, "direction": (raw / norm).astype(np.float32),
            "norm_raw": norm, "n_pairs": len(pairs),
            "n_incidentals": len({p["incidental_id"] for p, _ in pairs})}


def projections(direction: np.ndarray, rows: list[dict[str, Any]],
                get_state: StateFn) -> np.ndarray:
    return np.array([
        float(get_state(r["item_id"]).astype(np.float64) @ direction)
        for r in rows])


def decoder_eval(direction: np.ndarray, rows: list[dict[str, Any]],
                 get_state: StateFn, *, split: str) -> dict[str, Any]:
    """E11: fixed projection onto the train-fitted d; Pearson r against
    signed strength; AUC on |s| >= 1 rows only."""
    use = [r for r in rows if r["incidental_split"] == split]
    if len(use) < 3:
        return {"corr": float("nan"), "auc": float("nan"), "n": len(use)}
    proj = projections(direction, use, get_state)
    s = np.array([float(r["context_strength"]) for r in use])
    corr = float(np.corrcoef(proj, s)[0, 1]) if np.std(s) > 0 else float("nan")
    mask = np.abs(s) >= 1
    auc = float("nan")
    if mask.sum() >= 4:
        p, lab = proj[mask], (s[mask] > 0).astype(int)
        pos, neg = p[lab == 1], p[lab == 0]
        if len(pos) and len(neg):
            auc = float((
                (pos[:, None] > neg[None, :]).sum()
                + 0.5 * (pos[:, None] == neg[None, :]).sum())
                / (len(pos) * len(neg)))
    return {"corr": corr, "auc": auc, "n": len(use)}


def _states_matrix(rows: list[dict[str, Any]],
                   get_state: StateFn) -> np.ndarray:
    return np.stack([get_state(r["item_id"]).astype(np.float64)
                     for r in rows])


def _fit_from_matrix(rows: list[dict[str, Any]], S: np.ndarray,
                     strengths: np.ndarray,
                     train_mask: np.ndarray) -> np.ndarray | None:
    """Matrix fast path: the matched paired-difference mean equals
    mean(states of paired A-favor rows) - mean(states of paired B-favor
    rows); pairing recomputed from the (possibly permuted) strengths."""
    plus: dict[tuple, int] = {}
    minus: dict[tuple, int] = {}
    for i, r in enumerate(rows):
        if not train_mask[i]:
            continue
        s = int(strengths[i])
        if s == 0:
            continue
        key = (r["incidental_id"], r["display_order"], r["code_map_index"],
               r.get("paraphrase_id", 0), abs(s), r.get("codebook_pair_id"))
        (plus if s > 0 else minus)[key] = i
    common = sorted(plus.keys() & minus.keys(), key=str)
    if not common:
        return None
    pi = np.array([plus[k] for k in common])
    mi = np.array([minus[k] for k in common])
    raw = S[pi].mean(axis=0) - S[mi].mean(axis=0)
    norm = np.linalg.norm(raw)
    if not np.isfinite(norm) or norm == 0:
        return None
    return raw / norm


def _eval_from_matrix(d: np.ndarray, S: np.ndarray, strengths: np.ndarray,
                      eval_mask: np.ndarray) -> tuple[float, float]:
    proj = S[eval_mask] @ d
    s = strengths[eval_mask].astype(np.float64)
    corr = float(np.corrcoef(proj, s)[0, 1]) if np.std(s) > 0 else float("nan")
    mask = np.abs(s) >= 1
    auc = float("nan")
    if mask.sum() >= 4:
        p, lab = proj[mask], (s[mask] > 0)
        pos, neg = p[lab], p[~lab]
        if len(pos) and len(neg):
            auc = float(((pos[:, None] > neg[None, :]).sum()
                         + 0.5 * (pos[:, None] == neg[None, :]).sum())
                        / (len(pos) * len(neg)))
    return corr, auc


def permutation_bands(rows: list[dict[str, Any]], get_state: StateFn,
                      *, eval_split: str, seed_key: str,
                      n_perm: int = N_PERMUTATIONS) -> dict[str, float]:
    """95th pct of |corr| and AUC over strength-label permutations WITHIN
    incidental (E11), refitting the direction per permuted labeling."""
    rng = np.random.default_rng(stable_seed("permband", seed_key,
                                            base=SEED_BASE))
    S = _states_matrix(rows, get_state)
    base_s = np.array([int(r["context_strength"]) for r in rows])
    train_mask = np.array([r["incidental_split"] == "train" for r in rows])
    eval_mask = np.array([r["incidental_split"] == eval_split for r in rows])
    by_inc: dict[str, list[int]] = {}
    for idx, r in enumerate(rows):
        by_inc.setdefault(r["incidental_id"], []).append(idx)
    inc_idx = [np.array(v) for v in by_inc.values()]
    corrs, aucs = [], []
    for _ in range(n_perm):
        perm_s = base_s.copy()
        for idxs in inc_idx:
            perm_s[idxs] = perm_s[idxs[rng.permutation(len(idxs))]]
        d = _fit_from_matrix(rows, S, perm_s, train_mask)
        if d is None:
            continue
        corr, auc = _eval_from_matrix(d, S, perm_s, eval_mask)
        if np.isfinite(corr):
            corrs.append(abs(corr))
        if np.isfinite(auc):
            aucs.append(max(auc, 1 - auc))
    return {
        "corr_p95": float(np.quantile(corrs, 0.95)) if corrs else float("nan"),
        "auc_p95": float(np.quantile(aucs, 0.95)) if aucs else float("nan"),
        "n_perm_effective": len(corrs),
    }


def direction_stability(rows: list[dict[str, Any]], get_state: StateFn,
                        *, seed_key: str,
                        n_boot: int = N_BOOT_DIRECTIONS) -> dict[str, Any]:
    """§36.3: bootstrap split-half cosines over train incidentals + sign
    stability across codebook pairs and paraphrase families."""
    rng = np.random.default_rng(stable_seed("stab", seed_key,
                                            base=SEED_BASE))
    train = [r for r in rows if r["incidental_split"] == "train"]
    incs = sorted({r["incidental_id"] for r in train})
    S = _states_matrix(train, get_state)
    strengths = np.array([int(r["context_strength"]) for r in train])
    cosines = []
    for _ in range(n_boot):
        perm = rng.permutation(incs)
        half_a = set(perm[: len(incs) // 2])
        mask_a = np.array([r["incidental_id"] in half_a for r in train])
        da = _fit_from_matrix(train, S, strengths, mask_a)
        db = _fit_from_matrix(train, S, strengths, ~mask_a)
        if da is not None and db is not None:
            cosines.append(float(da @ db))
    full = fit_direction(train, get_state)
    strata_cos = {}
    if full["ok"]:
        for field in ("codebook_pair_id", "paraphrase_id"):
            for val in sorted({str(r.get(field)) for r in train}):
                sub = [r for r in train if str(r.get(field)) == val]
                f = fit_direction(sub, get_state)
                if f["ok"]:
                    strata_cos[f"{field}={val}"] = float(
                        f["direction"] @ full["direction"])
    cos = np.array(cosines) if cosines else np.array([np.nan])
    return {
        "median_split_half_cos": float(np.nanmedian(cos)),
        "p10_split_half_cos": float(np.nanquantile(cos, 0.10)),
        "strata_cosines": strata_cos,
        "strata_sign_stable": bool(strata_cos
                                   and all(v > 0 for v in strata_cos.values())),
    }


def neutral_relevance(direction: np.ndarray, rows: list[dict[str, Any]],
                      get_state: StateFn, *, split: str,
                      seed_key: str,
                      n_rand: int = 200) -> dict[str, Any]:
    """§36.4: projection predicts the neutral semantic margin, beating a
    matched random-direction band; survives the heldout codebook."""
    neu = [r for r in rows if r["incidental_split"] == split
           and int(r["context_strength"]) == 0]
    if len(neu) < 3:
        return {"corr": float("nan"), "rand_p95": float("nan")}
    proj = projections(direction, neu, get_state)
    marg = np.array([float(r["margin_full_a_minus_b"]) for r in neu])
    corr = (float(np.corrcoef(proj, marg)[0, 1])
            if np.std(marg) > 0 else float("nan"))
    rng = np.random.default_rng(stable_seed("neurand", seed_key,
                                            base=SEED_BASE))
    dim = len(direction)
    rand_corrs = []
    for _ in range(n_rand):
        v = rng.standard_normal(dim)
        v /= np.linalg.norm(v)
        p = projections(v.astype(np.float32), neu, get_state)
        if np.std(p) > 0 and np.std(marg) > 0:
            rand_corrs.append(abs(float(np.corrcoef(p, marg)[0, 1])))
    rand_p95 = (float(np.quantile(rand_corrs, 0.95))
                if rand_corrs else float("nan"))
    reserved = [r for r in neu if r.get("codebook_reserved")]
    primary = [r for r in neu if not r.get("codebook_reserved")]
    corr_reserved = float("nan")
    if len(reserved) >= 3:
        pr = projections(direction, reserved, get_state)
        mr = np.array([float(r["margin_full_a_minus_b"]) for r in reserved])
        if np.std(mr) > 0:
            corr_reserved = float(np.corrcoef(pr, mr)[0, 1])
    return {"corr": corr, "rand_p95": rand_p95,
            "corr_reserved_codebook": corr_reserved,
            "n_neutral": len(neu), "n_reserved": len(reserved)}


def identifiability_precheck(rows: list[dict[str, Any]], get_state: StateFn,
                             *, scenario_id: str, site: str, depth: int,
                             behavioral_slope_passes: bool) -> dict[str, Any]:
    """§36 full precheck for one scenario/site/depth cell (holdout is
    evaluated ONLY when the gate later opens; here validation governs)."""
    fit = fit_direction(rows, get_state)
    if not fit["ok"]:
        return {"ready": False, "reason": fit.get("reason"),
                "scenario_id": scenario_id, "site": site, "depth": depth}
    d = fit["direction"]
    seed_key = f"{scenario_id}|{site}|{depth}"
    val = decoder_eval(d, rows, get_state, split="validation")
    bands = permutation_bands(rows, get_state, eval_split="validation",
                              seed_key=seed_key)
    stab = direction_stability(rows, get_state, seed_key=seed_key)
    neu = neutral_relevance(d, rows, get_state, split="validation",
                            seed_key=seed_key)
    incs = {s: len({r["incidental_id"] for r in rows
                    if r["incidental_split"] == s})
            for s in ("train", "validation", "holdout")}
    checks = {
        "behavioral_slope": bool(behavioral_slope_passes),
        "decoder_corr": bool(np.isfinite(val["corr"])
                             and val["corr"] >= DECODER_CORR_MIN),
        "decoder_auc": bool(np.isfinite(val["auc"])
                            and val["auc"] >= DECODER_AUC_MIN),
        "corr_beats_permutation": bool(
            np.isfinite(val["corr"]) and np.isfinite(bands["corr_p95"])
            and abs(val["corr"]) > bands["corr_p95"]),
        "auc_beats_permutation": bool(
            np.isfinite(val["auc"]) and np.isfinite(bands["auc_p95"])
            and val["auc"] > bands["auc_p95"]),
        "split_half_median": bool(
            stab["median_split_half_cos"] >= SPLIT_HALF_COS_MEDIAN_MIN),
        "split_half_p10_positive": bool(stab["p10_split_half_cos"] > 0),
        "strata_sign_stable": bool(stab["strata_sign_stable"]),
        "neutral_corr_positive_beats_band": bool(
            np.isfinite(neu["corr"]) and neu["corr"] > 0
            and np.isfinite(neu["rand_p95"])
            and abs(neu["corr"]) > neu["rand_p95"]),
        "neutral_survives_reserved_codebook": bool(
            not np.isfinite(neu["corr_reserved_codebook"])
            or neu["corr_reserved_codebook"] > 0),
        "split_sizes": bool(incs["train"] >= MIN_TRAIN_INC
                            and incs["validation"] >= MIN_VAL_INC
                            and incs["holdout"] >= MIN_HOLDOUT_INC),
    }
    return {
        "scenario_id": scenario_id, "site": site, "depth": depth,
        "ready": all(checks.values()), "checks": checks,
        "fit": {"n_pairs": fit["n_pairs"], "norm_raw": fit["norm_raw"]},
        "validation_eval": val, "permutation_bands": bands,
        "stability": stab, "neutral_relevance": neu,
        "incidental_counts": incs,
    }


def _normalize_scores(cells: list[dict[str, Any]]) -> None:
    """Frozen normalization transforms (§37): decoder corr and neutral
    corr enter raw in [0,1] via clip; AUC gap = auc - band; split-half
    median in [0,1] via clip; cross-codebook = min stratum cosine
    clipped."""
    for c in cells:
        val, bands = c["validation_eval"], c["permutation_bands"]
        stab, neu = c["stability"], c["neutral_relevance"]
        strata = stab.get("strata_cosines") or {}
        c["score_terms"] = {
            "decoder_corr": float(np.clip(val["corr"], 0, 1))
            if np.isfinite(val["corr"]) else 0.0,
            "auc_gap": float(np.clip((val["auc"] or 0)
                                     - (bands["auc_p95"] or 0.5), 0, 0.5)) * 2
            if np.isfinite(val["auc"]) and np.isfinite(bands["auc_p95"])
            else 0.0,
            "split_half": float(np.clip(stab["median_split_half_cos"], 0, 1)),
            "neutral_corr": float(np.clip(neu["corr"], 0, 1))
            if np.isfinite(neu["corr"]) else 0.0,
            "cross_codebook": float(np.clip(min(strata.values()), 0, 1))
            if strata else 0.0,
        }
        c["score"] = float(sum(SCORE_WEIGHTS[k] * v
                               for k, v in c["score_terms"].items()))


def select_site_depth(cells: list[dict[str, Any]]) -> dict[str, Any] | None:
    """§37: highest validation-only score among ready cells; tie-break
    earlier upstream site, then shallower depth.

    Scores are quantized to 0.01 before comparison (frozen transform):
    without quantization, per-cell permutation-band seeds produce float
    jitter that would never let the declared upstream tie-break engage —
    the plan's stated intent is to favor propagation evidence over output
    adjacency among comparably scoring cells."""
    ready = [c for c in cells if c["ready"]]
    if not ready:
        return None
    _normalize_scores(ready)
    def key(c):
        return (-round(c["score"], 2), SITE_ORDER.index(c["site"]),
                c["depth"])
    return sorted(ready, key=key)[0]
