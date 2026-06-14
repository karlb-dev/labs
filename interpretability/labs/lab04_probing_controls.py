"""Lab 4: Probing without fooling yourself, now featuring truth.

This lab targets the DECODE rung of the evidence ladder (building directly on
the residual-stream indexing and "readout is an instrument" caution from Lab 1,
the frozen-norm discipline from Lab 2, and the routing-vs-contribution
distinction from Lab 3). It asks what is linearly decodable from the residual
stream, including statement truth, while keeping every answer glued to controls.

Two tracks are measured from the same cached forward passes (pre-final-norm
residuals at the final token position, exactly as in Labs 1-3):

* Surface track: a deliberately shallow final-word letter feature. This gives
  students a calibration curve for "trivially decodable" information on the
  same activations.
* Truth track: true/false statement labels from four frozen families in
  data/: cities, comparisons, negations, and misconceptions. Negations and
  misconceptions are stress tests, not saved-direction selectors.

Two probes are fit at every residual-stream depth:

* logistic regression, which can find any separating direction available to a
  linear classifier;
* mass-mean, the difference of class means, which is the direction saved for
  Lab 7's causal steering test (the bridge that turns "decodable" into a
  testable hypothesis about use).

The lab's product is not a single accuracy number. It is a skepticism packet:
shuffled-label refits, random-direction controls, length and majority
baselines, grouped train/eval splits (to prevent template leakage), family-
held-out transfer (including negation inversion), calibration, activation-norm
diagnostics (the "outlier specimen" that can hijack a mean), and a saved truth
direction with a readable card.

Evidence level: DECODE. Nothing here shows that the model *uses* the direction.
Lab 7 cashes that check with interventions on the saved mass-mean vector.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import math
import re
import statistics
from collections import defaultdict
from typing import Any, Iterable

import interp_bench as bench

LAB_ID = "L04"

FAMILIES = ("cities", "comparisons", "negations", "misconceptions")
# misconceptions stays OUT of AFFIRMATIVE_FAMILIES on purpose: it is the
# stress-test column of the generalization matrix (popular false beliefs whose
# text frequency anti-correlates with truth), not a driver of layer selection
# or of the saved truth_direction.pt that Lab 7 reuses. A probe that tracks
# assertion frequency rather than truth fails on this family — report it.
AFFIRMATIVE_FAMILIES = ("cities", "comparisons")
SAVED_DIRECTION_EVAL_FAMILIES = ("cities", "comparisons", "negations")
DATA_FILES = {
    "cities": "truth_cities.csv",
    "comparisons": "truth_comparisons.csv",
    "negations": "truth_negations.csv",
    "misconceptions": "truth_misconceptions.csv",
}
TRAIN_FRACTION = 0.7
N_SHUFFLES = 2          # shuffled-label refits per (depth, family, method)
N_RANDOM_DIRS = 3       # random-direction baselines per depth
LOGISTIC_L2 = 1e-2
SURFACE_LETTERS = tuple("abcdefghijklmnopqrstuvwxyz")  # candidate letters; most balanced wins
NORMALIZE_ROWS = True   # flip this for the outlier exercise in the handout
N_CALIBRATION_BINS = 8
OUTLIER_MULTIPLIER = 3.0


@dataclasses.dataclass(frozen=True)
class Statement:
    statement_id: str
    family: str
    statement: str
    label: int           # 1 = true
    meta: str = ""


@dataclasses.dataclass(frozen=True)
class SplitInfo:
    statement_id: str
    family: str
    split_key: str
    split: str           # train or eval


# ---------------------------------------------------------------------------
# Data loading and split hygiene
# ---------------------------------------------------------------------------


def stable_hash_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def data_file_digest(family: str) -> str | None:
    path = bench.COURSE_ROOT / "data" / DATA_FILES[family]
    return bench.sha256_file(path) if path.exists() else None


def load_family(family: str) -> list[Statement]:
    path = bench.COURSE_ROOT / "data" / DATA_FILES[family]
    if not path.exists():
        raise RuntimeError(
            f"Frozen dataset missing: {path}. The truth CSVs are vendored in "
            "data/. Re-checkout the repo; do not regenerate per-run."
        )
    out: list[Statement] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = int(row["label"])
            row_family = row.get("family", family)
            if row_family != family:
                raise RuntimeError(
                    f"Dataset row {row.get('statement_id')} declares family {row_family!r}, "
                    f"but it was loaded from {family!r}."
                )
            if label not in (0, 1):
                raise RuntimeError(f"Bad truth label for {row.get('statement_id')}: {label!r}")
            out.append(
                Statement(
                    statement_id=row["statement_id"],
                    family=row_family,
                    statement=row["statement"],
                    label=label,
                    meta=row.get("meta", ""),
                )
            )
    if not out:
        raise RuntimeError(f"Frozen dataset is empty: {path}")
    return out


def cap_balanced(statements: list[Statement], cap: int) -> list[Statement]:
    """Cap one family while keeping the true/false balance exact.

    The cap is rounded down to an even number. That is intentional: probe
    controls are more valuable than squeezing one more unbalanced example into
    a smoke run.
    """
    if cap <= 0 or cap >= len(statements):
        return statements
    true = [s for s in statements if s.label == 1]
    false = [s for s in statements if s.label == 0]
    half = min(len(true), len(false), max(1, cap // 2))
    out: list[Statement] = []
    for t, f in zip(true[:half], false[:half]):
        out.extend([t, f])
    return out


def _clean_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[.?!]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def split_group_key(statement: Statement) -> str:
    """Return a leakage-resistant grouping key for train/eval splitting.

    Statement-level hashing is too permissive for truth datasets. If
    "Paris is in France" is in train and "Paris is in the Netherlands" is in
    eval, a probe can ride entity/template structure. The split therefore
    groups obvious paired variants before assigning train or eval.
    """
    text = _clean_text(statement.statement)

    # Cities and negations. Group by city, not by country or negation token.
    m = re.match(r"^the city of (.+?) is (?:not )?in (.+)$", text)
    if m:
        city = re.sub(r"\s+", "_", m.group(1))
        return f"{statement.family}:city:{city}"

    # Numeric comparisons. Group by unordered pair of compared quantities.
    m = re.match(
        r"^(.+?) is (?:larger|greater|bigger|smaller|less|lower) than (.+)$",
        text,
    )
    if m:
        a = re.sub(r"\s+", "_", m.group(1))
        b = re.sub(r"\s+", "_", m.group(2))
        return f"{statement.family}:comparison:{'|'.join(sorted((a, b)))}"

    # Metadata is often a pair/group identifier in frozen eval CSVs, but do
    # not trust it blindly if it is empty or label-looking.
    meta = _clean_text(statement.meta)
    if meta and meta not in {"true", "false", "1", "0"}:
        return f"{statement.family}:meta:{meta}"

    # Last-resort fallback: strip common label suffixes from statement_id.
    key = statement.statement_id.lower()
    key = re.sub(r"([_-](true|false|t|f|yes|no|1|0))+$", "", key)
    return f"{statement.family}:id:{key}"


def _label_set(indices: Iterable[int], statements: list[Statement]) -> set[int]:
    return {statements[i].label for i in indices}


def make_grouped_split(statements: list[Statement]) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    """Assign train/eval splits by leakage-resistant groups.

    Returns a statement_id -> is_train map and group-level audit rows. The
    split is deterministic, grouped, and repaired so every family has both
    classes in both train and eval when the data makes that possible.
    """
    groups_by_family: dict[str, dict[str, list[int]]] = {f: defaultdict(list) for f in FAMILIES}
    for i, s in enumerate(statements):
        groups_by_family[s.family][split_group_key(s)].append(i)

    split_lookup: dict[str, bool] = {}
    audit_rows: list[dict[str, Any]] = []

    for family in FAMILIES:
        groups = groups_by_family[family]
        keys = sorted(groups, key=lambda k: stable_hash_int(k))
        if len(keys) < 2:
            raise RuntimeError(
                f"Family {family!r} has only {len(keys)} split group after grouping. "
                "Use a larger prompt set or inspect split_group_key()."
            )
        n_train_groups = int(round(TRAIN_FRACTION * len(keys)))
        n_train_groups = min(max(1, n_train_groups), len(keys) - 1)
        train_keys = set(keys[:n_train_groups])

        def split_indices(train: set[str]) -> tuple[list[int], list[int]]:
            train_idx = [i for k in train for i in groups[k]]
            eval_idx = [i for k in keys if k not in train for i in groups[k]]
            return train_idx, eval_idx

        def ok(train: set[str]) -> bool:
            tr, ev = split_indices(train)
            return _label_set(tr, statements) == {0, 1} and _label_set(ev, statements) == {0, 1}

        # Deterministic repair if the first hash split strands a class.
        for _ in range(len(keys) * 4):
            if ok(train_keys):
                break
            tr, ev = split_indices(train_keys)
            train_labels = _label_set(tr, statements)
            eval_labels = _label_set(ev, statements)
            moved = False
            for label in (0, 1):
                if label not in train_labels:
                    for k in keys:
                        if k not in train_keys and any(statements[i].label == label for i in groups[k]):
                            train_keys.add(k)
                            moved = True
                            break
                if moved:
                    break
                if label not in eval_labels:
                    for k in keys:
                        if k in train_keys and len(train_keys) > 1 and any(
                            statements[i].label == label for i in groups[k]
                        ):
                            train_keys.remove(k)
                            moved = True
                            break
                if moved:
                    break
            if not moved:
                break
        if not ok(train_keys):
            raise RuntimeError(
                f"Could not create a grouped train/eval split for {family!r} with both labels "
                "in both splits. Increase --max-examples or inspect diagnostics/split_audit.csv."
            )

        for k in keys:
            split = "train" if k in train_keys else "eval"
            idxs = groups[k]
            for i in idxs:
                split_lookup[statements[i].statement_id] = split == "train"
            audit_rows.append(
                {
                    "family": family,
                    "split_key": k,
                    "split": split,
                    "n_statements": len(idxs),
                    "n_true": sum(statements[i].label for i in idxs),
                    "n_false": sum(1 - statements[i].label for i in idxs),
                    "example_statement_ids": ";".join(statements[i].statement_id for i in idxs[:4]),
                }
            )
    return split_lookup, audit_rows


# ---------------------------------------------------------------------------
# Probes, controls, and calibration
# ---------------------------------------------------------------------------


def require_two_classes(y: Any, *, context: str) -> None:
    if not bool((y == 1).any()) or not bool((y == 0).any()):
        raise ValueError(f"{context} needs both classes; got labels {sorted(set(y.tolist()))}")


def labels_have_two_classes(y: Any) -> bool:
    """Guard shallow controls in tiny smoke splits, where a letter feature can be one-class."""
    try:
        vals = set(int(v) for v in y.tolist())
    except AttributeError:
        vals = set(int(v) for v in y)
    return vals == {0, 1}


def fit_logistic(X: Any, y: Any, l2: float = LOGISTIC_L2) -> dict[str, Any]:
    """L2-regularized logistic regression via torch LBFGS.

    Eval data is standardized with train-set statistics only (see the mu/sigma
    saved in the probe dict). Leaking eval-set scale into a probe is a quiet
    leakage bug that makes "decodability" look stronger than it is. This is the
    same "instrument hygiene" spirit as the frozen-norm linearization in Lab 2.
    """
    import torch

    require_two_classes(y, context="logistic probe")
    mu = X.mean(dim=0)
    sigma = X.std(dim=0).clamp_min(1e-6)
    Xs = (X - mu) / sigma
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=60, line_search_fn="strong_wolfe")
    yf = y.float()

    def closure():
        opt.zero_grad()
        logits = Xs @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yf)
        loss = loss + l2 * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    return {"w": w.detach(), "b": b.detach(), "mu": mu, "sigma": sigma}


def logistic_logits(probe: dict[str, Any], X: Any) -> Any:
    Xs = (X - probe["mu"]) / probe["sigma"]
    return Xs @ probe["w"] + probe["b"]


def logistic_probs(probe: dict[str, Any], X: Any) -> Any:
    import torch

    return torch.sigmoid(logistic_logits(probe, X))


def eval_logistic(probe: dict[str, Any], X: Any, y: Any) -> float:
    pred = logistic_logits(probe, X) > 0
    return float((pred == y.bool()).float().mean())


def eval_logistic_metrics(probe: dict[str, Any], X: Any, y: Any) -> dict[str, float]:
    import torch

    probs = logistic_probs(probe, X).detach().float().clamp(1e-6, 1 - 1e-6)
    yf = y.float()
    pred = probs > 0.5
    brier = torch.mean((probs - yf) ** 2)
    nll = torch.nn.functional.binary_cross_entropy(probs, yf)
    return {
        "accuracy": float((pred == y.bool()).float().mean()),
        "brier": float(brier),
        "nll": float(nll),
        "ece": expected_calibration_error(probs, y),
    }


def fit_mass_mean(X: Any, y: Any) -> dict[str, Any]:
    """Difference of class means; threshold at the projected midpoint.

    This is the probe whose direction is saved for Lab 7's causal test. It is
    deliberately simpler than logistic regression (difference of means rather
    than max-margin separator) so that an intervention ("add this vector at
    this scale") has a clean geometric meaning. The disagreement between
    logistic and mass-mean is data, not noise — it is one of the lab's core
    artifacts (see the decodability plot and the truth_projection_panels).
    """
    require_two_classes(y, context="mass-mean probe")
    mu_true = X[y == 1].mean(dim=0)
    mu_false = X[y == 0].mean(dim=0)
    direction = mu_true - mu_false
    threshold = float(((mu_true + mu_false) / 2) @ direction)
    return {"direction": direction, "threshold": threshold}


def eval_mass_mean(probe: dict[str, Any], X: Any, y: Any) -> float:
    pred = (X @ probe["direction"]) > probe["threshold"]
    return float((pred == y.bool()).float().mean())


def eval_random_directions(X_train: Any, y_train: Any, X_eval: Any, y_eval: Any, seed: int) -> float:
    """Random unit vectors with midpoint thresholds fit on train."""
    import torch

    require_two_classes(y_train, context="random-direction control")
    accs = []
    gen = torch.Generator().manual_seed(seed)
    for _ in range(N_RANDOM_DIRS):
        d = torch.randn(X_train.shape[1], generator=gen)
        d = d / d.norm().clamp_min(1e-9)
        mu_t = X_train[y_train == 1] @ d
        mu_f = X_train[y_train == 0] @ d
        thr = float((mu_t.mean() + mu_f.mean()) / 2)
        sign = 1.0 if float(mu_t.mean()) >= float(mu_f.mean()) else -1.0
        pred = (sign * (X_eval @ d)) > sign * thr
        accs.append(float((pred == y_eval.bool()).float().mean()))
    return statistics.fmean(accs)


def shuffled_labels(y: Any, seed: int) -> Any:
    import torch

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(y), generator=gen)
    return y[perm]


def majority_baseline(y_train: Any, y_eval: Any) -> float:
    pred = bool(float(y_train.float().mean()) >= 0.5)
    return float((y_eval.bool() == pred).float().mean())


def expected_calibration_error(probs: Any, y: Any, n_bins: int = N_CALIBRATION_BINS) -> float:
    import torch

    probs = probs.detach().float()
    labels = y.float()
    ece = torch.tensor(0.0)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (probs >= lo) & (probs < hi if b < n_bins - 1 else probs <= hi)
        if bool(mask.any()):
            conf = probs[mask].mean()
            acc = labels[mask].mean()
            ece = ece + mask.float().mean() * torch.abs(conf - acc)
    return float(ece)


def calibration_bins(probs: Any, y: Any, *, family: str, layer: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    probs = probs.detach().float()
    labels = y.float()
    for b in range(N_CALIBRATION_BINS):
        lo, hi = b / N_CALIBRATION_BINS, (b + 1) / N_CALIBRATION_BINS
        mask = (probs >= lo) & (probs < hi if b < N_CALIBRATION_BINS - 1 else probs <= hi)
        n = int(mask.sum())
        rows.append(
            {
                "family": family,
                "layer": layer,
                "bin": b,
                "prob_lo": round(lo, 4),
                "prob_hi": round(hi, 4),
                "n": n,
                "mean_predicted_prob": round(float(probs[mask].mean()), 4) if n else "",
                "empirical_true_rate": round(float(labels[mask].mean()), 4) if n else "",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Surface track feature
# ---------------------------------------------------------------------------


def final_word(statement: str) -> str:
    return statement.rstrip(".?!").split()[-1]


def pick_surface_letter(statements: list[Statement], split_lookup: dict[str, bool] | None = None) -> str:
    """Choose a final-word letter that is balanced and trainable.

    The surface track is a calibration curve, not a puzzle box. We therefore
    prefer a letter that gives both classes inside every family split. If no
    candidate satisfies that strict criterion, the score degrades gracefully
    toward global balance.
    """
    best, best_score = SURFACE_LETTERS[0], float("inf")
    for letter in SURFACE_LETTERS:
        vals = [1 if letter in final_word(s.statement).lower() else 0 for s in statements]
        frac = statistics.fmean(vals)
        score = abs(frac - 0.5)
        if split_lookup is not None:
            for family in FAMILIES:
                for is_train_split in (True, False):
                    split_vals = [
                        1 if letter in final_word(s.statement).lower() else 0
                        for s in statements
                        if s.family == family and split_lookup[s.statement_id] == is_train_split
                    ]
                    # A one-class split would make the surface probe either
                    # fail or become a majority baseline in costume. Penalize it.
                    if len(set(split_vals)) < 2:
                        score += 10.0
                    else:
                        score += 0.1 * abs(statistics.fmean(split_vals) - 0.5)
        if score < best_score:
            best, best_score = letter, score
    return best


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

# Lab 4 is where “a probe found it” goes on trial.  The plotting grammar uses
# the same visual rails everywhere: red = truth probe, purple = saved mass-mean
# direction, gray/brown/black = controls, and family colors stay stable across
# matrices, strips, and dashboards.
FAMILY_COLORS = {
    "cities": "#0072B2",          # blue
    "comparisons": "#E69F00",    # orange
    "negations": "#009E73",      # green
    "misconceptions": "#D55E00", # vermillion
}
FAMILY_MARKERS = {"cities": "o", "comparisons": "s", "negations": "^", "misconceptions": "D"}
METHOD_COLORS = {"logistic": "#D55E00", "mass_mean": "#7E57C2", "surface": "#666666"}
CONTROL_COLORS = {
    "real": "#D55E00",
    "shuffled": "#222222",
    "random": "#8C564B",
    "surface": "#666666",
    "length": "#56B4E9",
    "majority": "#999999",
    "mass_mean": "#7E57C2",
}


def family_color(family: str) -> str:
    """Color for a statement family; unknown families get a stable fallback
    instead of a KeyError (adding a CSV family must never crash a plot)."""
    if hasattr(bench, "plot_category_color"):
        return bench.plot_category_color(family, FAMILY_COLORS.get(family, "#555555"))
    return FAMILY_COLORS.get(family, "#555555")


def family_marker(family: str) -> str:
    if hasattr(bench, "plot_category_marker"):
        return bench.plot_category_marker(family, FAMILY_MARKERS.get(family, "o"))
    return FAMILY_MARKERS.get(family, "o")


def _lighten(color: str, amount: float = 0.55) -> str:
    if hasattr(bench, "lighten_color"):
        return bench.lighten_color(color, amount)
    import matplotlib.colors as mcolors

    r, g, b = mcolors.to_rgb(color)
    return mcolors.to_hex((r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount))


def _style(ax: Any, *, title: str = "", xlabel: str = "", ylabel: str = "", legend: bool = True, legend_loc: str = "best") -> None:
    if hasattr(bench, "style_ax"):
        bench.style_ax(ax, title=title or None, xlabel=xlabel or None, ylabel=ylabel or None, legend=legend, legend_loc=legend_loc)
    else:
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend(loc=legend_loc, fontsize=8, frameon=False)
        ax.grid(True, alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)


def _panel(ax: Any, label: str) -> None:
    if hasattr(bench, "add_panel_label"):
        bench.add_panel_label(ax, label)
    else:
        ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="right", va="bottom")


def _label_end(ax: Any, xs: list[float], ys: list[float], label: str, color: str) -> None:
    if hasattr(bench, "label_line_end"):
        bench.label_line_end(ax, xs, ys, label, color=color)
        return
    for x, y in zip(reversed(xs), reversed(ys)):
        try:
            xf, yf = float(x), float(y)
        except (TypeError, ValueError):
            continue
        if math.isfinite(xf) and math.isfinite(yf):
            ax.annotate(label, (xf, yf), textcoords="offset points", xytext=(3, 0), va="center", fontsize=7.5, color=color)
            return


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _rows(report: list[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    out = []
    for r in report:
        ok = True
        for k, v in filters.items():
            if v is None:
                continue
            if r.get(k) != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def _acc(
    report: list[dict[str, Any]],
    *,
    track: str = "truth",
    method: str,
    eval_kind: str,
    layer: int,
    train_family: str | None = None,
    eval_family: str | None = None,
) -> float | None:
    vals = []
    for r in report:
        if r.get("track") != track or r.get("method") != method or r.get("eval_kind") != eval_kind or int(r.get("layer", -999)) != layer:
            continue
        if train_family is not None and r.get("train_family") != train_family:
            continue
        if eval_family is not None and r.get("eval_family") != eval_family:
            continue
        v = _float_or_none(r.get("accuracy"))
        if v is not None:
            vals.append(v)
    return statistics.fmean(vals) if vals else None


def _curve_summary(report: list[dict[str, Any]], predicate) -> tuple[list[int], list[float], list[float], list[float]]:
    import numpy as np

    by_layer: dict[int, list[float]] = {}
    for r in report:
        if predicate(r) and int(r.get("layer", -1)) >= 0:
            v = _float_or_none(r.get("accuracy"))
            if v is not None:
                by_layer.setdefault(int(r["layer"]), []).append(v)
    depths = sorted(by_layer)
    med, lo, hi = [], [], []
    for k in depths:
        arr = np.array(by_layer[k], dtype=float)
        med.append(float(np.median(arr)))
        lo.append(float(np.quantile(arr, 0.25)))
        hi.append(float(np.quantile(arr, 0.75)))
    return depths, med, lo, hi


def _plot_curve_with_band(ax: Any, depths: list[int], med: list[float], lo: list[float], hi: list[float], *, label: str, color: str, linestyle: str = "-", linewidth: float = 2.4) -> None:
    if not depths:
        return
    ax.fill_between(depths, lo, hi, color=_lighten(color, 0.72), alpha=0.45, linewidth=0)
    ax.plot(depths, med, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
    _label_end(ax, depths, med, label.split(",")[0], color)


def _mean_curve(report: list[dict[str, Any]], predicate) -> tuple[list[int], list[float]]:
    by_layer: dict[int, list[float]] = {}
    for r in report:
        if predicate(r) and int(r.get("layer", -1)) >= 0:
            v = _float_or_none(r.get("accuracy"))
            if v is not None:
                by_layer.setdefault(int(r["layer"]), []).append(v)
    depths = sorted(by_layer)
    return depths, [statistics.fmean(by_layer[k]) for k in depths]


def _selectivity_lookup(selectivity_rows: list[dict[str, Any]], family: str, method: str, layer: int) -> float | None:
    vals = [
        _float_or_none(r.get("selectivity"))
        for r in selectivity_rows
        if r.get("family") == family and r.get("method") == method and int(r.get("layer", -999)) == layer
    ]
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def _depth_extent(report: list[dict[str, Any]]) -> int:
    depths = [int(r["layer"]) for r in report if int(r.get("layer", -1)) >= 0]
    return max(depths) if depths else 1


def build_control_ladder_rows(report: list[dict[str, Any]], truth_peak_layer: int, surface_letter: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        real = _acc(report, method="logistic", eval_kind="within", layer=truth_peak_layer, train_family=family, eval_family=family)
        shuffled = _acc(report, method="logistic", eval_kind="shuffled_control", layer=truth_peak_layer, train_family=family, eval_family=family)
        random = _acc(report, method="random_direction", eval_kind="random_control", layer=truth_peak_layer, train_family=family, eval_family=family)
        surface = _acc(report, track="surface", method="logistic", eval_kind="within", layer=truth_peak_layer, train_family=family, eval_family=family)
        length = _acc(report, method="token_length_baseline", eval_kind="length_control", layer=-1, train_family=family, eval_family=family)
        majority = _acc(report, method="majority_baseline", eval_kind="majority_control", layer=-1, train_family=family, eval_family=family)
        rows.append(
            {
                "family": family,
                "layer": truth_peak_layer,
                "surface_feature": f"final word contains {surface_letter!r}",
                "truth_logistic_accuracy": "" if real is None else round(real, 4),
                "shuffled_control_accuracy": "" if shuffled is None else round(shuffled, 4),
                "random_direction_accuracy": "" if random is None else round(random, 4),
                "surface_accuracy": "" if surface is None else round(surface, 4),
                "token_length_baseline": "" if length is None else round(length, 4),
                "majority_baseline": "" if majority is None else round(majority, 4),
                "selectivity_vs_shuffled": "" if real is None or shuffled is None else round(real - shuffled, 4),
            }
        )
    return rows


def build_probe_evidence_rows(
    report: list[dict[str, Any]],
    selectivity_rows: list[dict[str, Any]],
    calibration_summary: list[dict[str, Any]],
    truth_peak_layer: int,
    best_layer: int,
) -> list[dict[str, Any]]:
    ece_by_family = {r["family"]: _float_or_none(r.get("ece")) for r in calibration_summary}
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        peak_logistic = _acc(report, method="logistic", eval_kind="within", layer=truth_peak_layer, train_family=family, eval_family=family)
        peak_shuffled = _acc(report, method="logistic", eval_kind="shuffled_control", layer=truth_peak_layer, train_family=family, eval_family=family)
        random_peak = _acc(report, method="random_direction", eval_kind="random_control", layer=truth_peak_layer, train_family=family, eval_family=family)
        saved_mass = _acc(report, method="mass_mean", eval_kind="within", layer=best_layer, train_family=family, eval_family=family)
        saved_mass_sel = _selectivity_lookup(selectivity_rows, family, "mass_mean", best_layer)
        cross_vals = [
            _float_or_none(r.get("accuracy"))
            for r in report
            if r.get("track") == "truth" and r.get("method") == "mass_mean" and int(r.get("layer", -1)) == best_layer
            and r.get("train_family") == family and r.get("eval_kind") == "cross"
        ]
        cross_vals = [v for v in cross_vals if v is not None]
        mis = _acc(report, method="mass_mean", eval_kind="cross" if family != "misconceptions" else "within", layer=best_layer, train_family=family, eval_family="misconceptions")
        surface = _acc(report, track="surface", method="logistic", eval_kind="within", layer=truth_peak_layer, train_family=family, eval_family=family)
        length = _acc(report, method="token_length_baseline", eval_kind="length_control", layer=-1, train_family=family, eval_family=family)
        rows.append(
            {
                "family": family,
                "truth_peak_layer": truth_peak_layer,
                "saved_direction_layer": best_layer,
                "logistic_accuracy_at_truth_peak": "" if peak_logistic is None else round(peak_logistic, 4),
                "logistic_shuffled_at_truth_peak": "" if peak_shuffled is None else round(peak_shuffled, 4),
                "logistic_selectivity_at_truth_peak": "" if peak_logistic is None or peak_shuffled is None else round(peak_logistic - peak_shuffled, 4),
                "random_direction_at_truth_peak": "" if random_peak is None else round(random_peak, 4),
                "mass_mean_accuracy_at_saved_layer": "" if saved_mass is None else round(saved_mass, 4),
                "mass_mean_selectivity_at_saved_layer": "" if saved_mass_sel is None else round(saved_mass_sel, 4),
                "minimum_mass_mean_cross_family_at_saved_layer": "" if not cross_vals else round(min(cross_vals), 4),
                "misconceptions_transfer_at_saved_layer": "" if mis is None else round(mis, 4),
                "surface_accuracy_at_truth_peak": "" if surface is None else round(surface, 4),
                "token_length_baseline": "" if length is None else round(length, 4),
                "calibration_ece_at_truth_peak": "" if ece_by_family.get(family) is None else round(float(ece_by_family[family]), 4),
            }
        )
    return rows


def lab4_plot_reading_guide() -> list[dict[str, str]]:
    return [
        {"plot": "probe_evidence_dashboard.png", "concept": "one-screen skepticism packet", "reading_question": "Does truth beat controls, transfer, and calibrate, or merely look accurate?"},
        {"plot": "decodability_by_layer.png", "concept": "decodability over depth", "reading_question": "When do real probes separate from shuffled/random/surface probes?"},
        {"plot": "family_depth_atlas.png", "concept": "family-specific emergence", "reading_question": "Which families become decodable early, late, or never?"},
        {"plot": "generalization_matrix.png", "concept": "held-out family transfer", "reading_question": "Does a direction trained on one family survive another, invert on negations, or fail on misconceptions?"},
        {"plot": "truth_direction_projection_strip.png", "concept": "saved vector geometry", "reading_question": "At the saved layer, do true and false statements fall on opposite sides of the mass-mean threshold?"},
        {"plot": "probe_control_ladder.png", "concept": "control ladder", "reading_question": "How much of the headline accuracy survives shuffled labels, random directions, length, majority, and surface controls?"},
        {"plot": "logistic_vs_massmean_gap.png", "concept": "flexible classifier vs simple direction", "reading_question": "Where does logistic regression succeed beyond the vector that Lab 7 will actually test?"},
        {"plot": "truth_calibration_curve.png", "concept": "confidence hygiene", "reading_question": "Does predicted P(true) mean what it says?"},
        {"plot": "norm_outlier_trajectories.png", "concept": "mean-direction fragility", "reading_question": "Which rows have enough norm mass to bend a difference-of-means direction?"},
        {"plot": "split_balance_audit.png", "concept": "leakage and balance hygiene", "reading_question": "Do train/eval splits preserve both labels inside every family?"},
    ]


def plot_decodability(
    ctx: bench.RunContext,
    report: list[dict[str, Any]],
    surface_letter: str,
    best_layer: int,
    truth_peak_layer: int,
) -> None:
    """Headline figure: truth, surface, and controls on one set of axes."""
    fig, ax = bench.new_figure(figsize=(11.2, 6.2))
    max_depth = _depth_extent(report)
    ax.axhspan(0.45, 0.55, color="#BBBBBB", alpha=0.18, label="chance neighborhood")
    curves = [
        (
            "truth, logistic within-family",
            lambda r: r["track"] == "truth" and r["method"] == "logistic" and r["eval_kind"] == "within",
            METHOD_COLORS["logistic"],
            "-",
            2.8,
        ),
        (
            "truth, mass-mean within-family",
            lambda r: r["track"] == "truth" and r["method"] == "mass_mean" and r["eval_kind"] == "within",
            METHOD_COLORS["mass_mean"],
            "--",
            2.7,
        ),
        (
            f"surface, final word contains {surface_letter!r}",
            lambda r: r["track"] == "surface" and r["method"] == "logistic" and r["eval_kind"] == "within",
            CONTROL_COLORS["surface"],
            "-",
            2.2,
        ),
        (
            "truth shuffled-label control",
            lambda r: r["track"] == "truth" and r["method"] == "logistic" and r["eval_kind"] == "shuffled_control",
            CONTROL_COLORS["shuffled"],
            ":",
            1.8,
        ),
        (
            "truth random-direction control",
            lambda r: r["track"] == "truth" and r["method"] == "random_direction",
            CONTROL_COLORS["random"],
            ":",
            1.8,
        ),
    ]
    for label, pred, color, linestyle, linewidth in curves:
        depths, med, lo, hi = _curve_summary(report, pred)
        _plot_curve_with_band(ax, depths, med, lo, hi, label=label, color=color, linestyle=linestyle, linewidth=linewidth)

    ax.axvline(best_layer, color=METHOD_COLORS["mass_mean"], linewidth=1.1, alpha=0.50, label=f"saved direction depth {best_layer}")
    if truth_peak_layer != best_layer:
        ax.axvline(truth_peak_layer, color=METHOD_COLORS["logistic"], linewidth=0.9, alpha=0.35, label=f"logistic peak depth {truth_peak_layer}")
    if hasattr(bench, "add_depth_phase_guides"):
        bench.add_depth_phase_guides(ax, max_depth, label_final=True)
    ax.set_xlim(0, max_depth + 1)
    ax.set_ylim(0.0, 1.03)
    _style(
        ax,
        title="Probe accuracy by depth: medians with family IQR bands",
        xlabel="residual-stream depth (0 = embeddings, k = after k blocks)",
        ylabel="held-out accuracy",
        legend=True,
        legend_loc="lower right",
    )
    bench.save_figure(
        ctx,
        fig,
        "decodability_by_layer.png",
        "Surface vs truth decodability by depth, with shuffled/random controls and family IQR bands.",
    )


def plot_generalization(ctx: bench.RunContext, report: list[dict[str, Any]], best_layer: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.9), constrained_layout=True)
    grids: dict[str, Any] = {}
    im = None
    for panel_i, (ax, method) in enumerate(zip(axes[:2], ("logistic", "mass_mean"))):
        grid = np.full((len(FAMILIES), len(FAMILIES)), np.nan)
        for r in report:
            if (
                r["track"] == "truth"
                and r["method"] == method
                and int(r["layer"]) == best_layer
                and r["eval_kind"] in ("within", "cross")
            ):
                i = FAMILIES.index(r["train_family"])
                j = FAMILIES.index(r["eval_family"])
                grid[i, j] = float(r["accuracy"])
        grids[method] = grid
        im = ax.imshow(grid, cmap="RdYlGn", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(FAMILIES)))
        ax.set_xticklabels(FAMILIES, rotation=25, ha="right")
        ax.set_yticks(range(len(FAMILIES)))
        ax.set_yticklabels(FAMILIES)
        ax.set_xlabel("evaluated on")
        ax.set_ylabel("trained on" if panel_i == 0 else "")
        ax.set_title(f"{method} @ depth {best_layer}")
        for i in range(len(FAMILIES)):
            for j in range(len(FAMILIES)):
                if not np.isnan(grid[i, j]):
                    weight = "bold" if grid[i, j] < 0.5 else "normal"
                    ax.annotate(f"{grid[i, j]:.2f}", (j, i), ha="center", va="center", fontsize=10, fontweight=weight)
        ax.tick_params(length=0)
    if "logistic" in grids and "mass_mean" in grids:
        delta = grids["logistic"] - grids["mass_mean"]
        ax = axes[2]
        im_delta = ax.imshow(delta, cmap="coolwarm", vmin=-0.5, vmax=0.5)
        ax.set_xticks(range(len(FAMILIES)))
        ax.set_xticklabels(FAMILIES, rotation=25, ha="right")
        ax.set_yticks(range(len(FAMILIES)))
        ax.set_yticklabels(FAMILIES)
        ax.set_xlabel("evaluated on")
        ax.set_title("logistic minus mass-mean")
        for i in range(len(FAMILIES)):
            for j in range(len(FAMILIES)):
                if not np.isnan(delta[i, j]):
                    ax.annotate(f"{delta[i, j]:+.2f}", (j, i), ha="center", va="center", fontsize=9)
        fig.colorbar(im_delta, ax=ax, fraction=0.046, label="accuracy gap")
    if im is not None:
        fig.colorbar(im, ax=axes[:2], fraction=0.026, label="accuracy")
    fig.suptitle("Family-held-out transfer: below chance can be structured anti-correlation")
    bench.save_figure(
        ctx,
        fig,
        "generalization_matrix.png",
        "Train-family x eval-family accuracy for both probe types at the saved direction depth, plus logistic-minus-mass-mean gap.",
    )


def plot_selectivity(ctx: bench.RunContext, selectivity_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), sharey=True, constrained_layout=True)
    for ax, method in zip(axes, ("logistic", "mass_mean")):
        for family in FAMILIES:
            rows = [r for r in selectivity_rows if r["family"] == family and r["method"] == method]
            rows = sorted(rows, key=lambda r: r["layer"])
            if rows:
                xs = [int(r["layer"]) for r in rows]
                ys = [float(r["selectivity"]) for r in rows]
                ax.plot(xs, ys, linewidth=2.0, color=family_color(family), label=family)
                if ys:
                    peak_i = max(range(len(ys)), key=lambda i: ys[i])
                    ax.scatter([xs[peak_i]], [ys[peak_i]], s=42, color=family_color(family), edgecolor="white", linewidth=0.7, zorder=5)
                    _label_end(ax, xs, ys, family, family_color(family))
        ax.axhline(0, color="black", linewidth=0.8)
        _style(
            ax,
            title=f"{method}: real minus shuffled",
            xlabel="residual-stream depth",
            ylabel="selectivity" if ax is axes[0] else "",
            legend=ax is axes[0],
            legend_loc="upper left",
        )
    fig.suptitle("Selectivity by family: raw accuracy only gets admitted after subtracting controls")
    bench.save_figure(ctx, fig, "selectivity_by_layer.png", "Per-family truth-probe selectivity by depth for logistic and mass-mean probes.")


def plot_projection_panels(
    ctx: bench.RunContext,
    X_layers: Any,            # [n, L+1, d]
    labels: Any,              # [n]
    families: list[str],
    n_depths: int,
    outlier_mask: Any = None,
) -> None:
    """Cities statements projected onto mass-mean dir plus top orthogonal PC."""
    import matplotlib.pyplot as plt
    import torch

    idx = [i for i, f in enumerate(families) if f == "cities" and (outlier_mask is None or not bool(outlier_mask[i]))]
    if len(idx) < 8:
        return
    Xc = X_layers[idx]
    yc = labels[idx]
    depths = sorted({1, n_depths // 4, n_depths // 2, (3 * n_depths) // 4, n_depths - 1})
    fig, axes = plt.subplots(1, len(depths), figsize=(3.55 * len(depths), 3.9), sharey=False, constrained_layout=True)
    if len(depths) == 1:
        axes = [axes]
    for ax, k in zip(axes, depths):
        X = Xc[:, k, :]
        mm = fit_mass_mean(X, yc)
        d1 = mm["direction"] / mm["direction"].norm().clamp_min(1e-9)
        Xp = X - X.mean(dim=0)
        resid = Xp - (Xp @ d1)[:, None] * d1[None, :]
        v = resid.T @ resid @ torch.ones(resid.shape[1]) / resid.shape[1]
        for _ in range(8):
            v = resid.T @ (resid @ v)
            v = v / v.norm().clamp_min(1e-9)
        x_proj = Xp @ d1
        y_proj = Xp @ v
        ax.axvline(0, color="#222222", linewidth=0.7, alpha=0.65)
        ax.scatter(x_proj[yc == 1], y_proj[yc == 1], s=28, color=family_color("cities"), alpha=0.82, label="true", edgecolor="white", linewidth=0.4)
        ax.scatter(x_proj[yc == 0], y_proj[yc == 0], s=28, color=family_color("misconceptions"), alpha=0.82, label="false", edgecolor="white", linewidth=0.4)
        _style(ax, title=f"depth {k}", xlabel="mass-mean direction", ylabel="top orthogonal PC" if ax is axes[0] else "", legend=ax is axes[0])
    fig.suptitle("Cities: truth separation over depth (raw-norm outliers hidden from view only)")
    bench.save_figure(ctx, fig, "truth_projection_panels.png", "2-D projections of city statements at five depths.")


def plot_norm_diagnostics(ctx: bench.RunContext, row_norms: Any, n_depths: int) -> None:
    import numpy as np

    norms = row_norms.numpy()
    depths = list(range(n_depths))
    med = np.median(norms, axis=0)
    p95 = np.quantile(norms, 0.95, axis=0)
    p99 = np.quantile(norms, 0.99, axis=0)
    mx = np.max(norms, axis=0)
    fig, ax = bench.new_figure(figsize=(10.0, 5.2))
    ax.plot(depths, med, linewidth=2.4, label="median", color="#0072B2")
    ax.fill_between(depths, med, p95, color=_lighten("#0072B2", 0.78), alpha=0.6, label="median → 95th percentile")
    ax.plot(depths, p99, linewidth=1.5, linestyle="--", label="99th percentile", color="#E69F00")
    ax.plot(depths, mx, linewidth=1.8, linestyle=":", label="max", color="#009E73")
    if float(mx[-1]) / max(float(med[-1]), 1e-9) > 8:
        ax.set_yscale("symlog", linthresh=max(float(med[-1]), 1.0))
    ax.set_xlabel("residual-stream depth")
    ax.set_ylabel("raw activation norm at final token")
    ax.set_title("Activation-norm diagnostics: the mean direction has mass, and mass has leverage")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "activation_norms_by_depth.png", "Median, p95, p99, and max final-token stream norms by depth.")


def plot_calibration(ctx: bench.RunContext, calibration_rows: list[dict[str, Any]], layer: int) -> None:
    fig, ax = bench.new_figure(figsize=(6.7, 5.7))
    ax.plot([0, 1], [0, 1], linestyle=":", color="black", linewidth=1.0, label="perfect calibration")
    for family in FAMILIES:
        rows = [r for r in calibration_rows if r["family"] == family and r["n"]]
        if not rows:
            continue
        xs = [float(r["mean_predicted_prob"]) for r in rows]
        ys = [float(r["empirical_true_rate"]) for r in rows]
        ns = [float(r["n"]) for r in rows]
        sizes = [24 + 7 * n for n in ns]
        ax.plot(xs, ys, linewidth=1.5, color=family_color(family), alpha=0.75)
        ax.scatter(xs, ys, s=sizes, color=family_color(family), alpha=0.78, label=family, edgecolor="white", linewidth=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted P(true)")
    ax.set_ylabel("empirical true rate")
    ax.set_title(f"Logistic truth-probe calibration at depth {layer}\npoint area = bin count")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "truth_calibration_curve.png", "Calibration curve for the logistic truth probe at its peak depth; bubble area encodes bin count.")


def plot_probe_dashboard(
    ctx: bench.RunContext,
    report: list[dict[str, Any]],
    selectivity_rows: list[dict[str, Any]],
    calibration_summary: list[dict[str, Any]],
    surface_letter: str,
    best_layer: int,
    truth_peak_layer: int,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.0), constrained_layout=True)
    ax = axes[0, 0]
    ax.axhspan(0.45, 0.55, color="#BBBBBB", alpha=0.18)
    specs = [
        ("truth logistic", lambda r: r["track"] == "truth" and r["method"] == "logistic" and r["eval_kind"] == "within", METHOD_COLORS["logistic"], "-"),
        ("truth mass-mean", lambda r: r["track"] == "truth" and r["method"] == "mass_mean" and r["eval_kind"] == "within", METHOD_COLORS["mass_mean"], "--"),
        ("surface", lambda r: r["track"] == "surface" and r["method"] == "logistic" and r["eval_kind"] == "within", CONTROL_COLORS["surface"], "-"),
        ("shuffled", lambda r: r["track"] == "truth" and r["method"] == "logistic" and r["eval_kind"] == "shuffled_control", CONTROL_COLORS["shuffled"], ":"),
    ]
    for label, pred, color, ls in specs:
        depths, med, lo, hi = _curve_summary(report, pred)
        _plot_curve_with_band(ax, depths, med, lo, hi, label=label, color=color, linestyle=ls, linewidth=2.2)
    ax.axvline(best_layer, color=METHOD_COLORS["mass_mean"], linewidth=1.0, alpha=0.45)
    ax.axvline(truth_peak_layer, color=METHOD_COLORS["logistic"], linewidth=0.9, alpha=0.35)
    _panel(ax, "A")
    _style(ax, title="Accuracy: medians + family IQR", xlabel="depth", ylabel="held-out accuracy", legend=True, legend_loc="lower right")

    ax = axes[0, 1]
    for family in FAMILIES:
        rows = sorted([r for r in selectivity_rows if r["family"] == family and r["method"] == "logistic"], key=lambda r: r["layer"])
        if rows:
            xs = [int(r["layer"]) for r in rows]
            ys = [float(r["selectivity"]) for r in rows]
            ax.plot(xs, ys, color=family_color(family), label=family, linewidth=2.0)
            _label_end(ax, xs, ys, family, family_color(family))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(truth_peak_layer, color=METHOD_COLORS["logistic"], linewidth=0.9, alpha=0.35)
    _panel(ax, "B")
    _style(ax, title="Selectivity: logistic minus shuffled", xlabel="depth", ylabel="accuracy gap", legend=True, legend_loc="upper left")

    ax = axes[1, 0]
    # At the saved layer, compare within-family mass-mean accuracy against worst cross-family transfer.
    x = np.arange(len(FAMILIES))
    width = 0.35
    within = []
    worst_cross = []
    for family in FAMILIES:
        w = _acc(report, method="mass_mean", eval_kind="within", layer=best_layer, train_family=family, eval_family=family)
        vals = [
            _float_or_none(r.get("accuracy"))
            for r in report
            if r.get("track") == "truth" and r.get("method") == "mass_mean" and int(r.get("layer", -1)) == best_layer
            and r.get("train_family") == family and r.get("eval_kind") == "cross"
        ]
        vals = [v for v in vals if v is not None]
        within.append(0.0 if w is None else w)
        worst_cross.append(0.0 if not vals else min(vals))
    ax.axhspan(0.45, 0.55, color="#BBBBBB", alpha=0.18)
    ax.bar(x - width / 2, within, width, label="within", color=METHOD_COLORS["mass_mean"], alpha=0.85)
    ax.bar(x + width / 2, worst_cross, width, label="worst cross", color=_lighten(METHOD_COLORS["mass_mean"], 0.45), alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(FAMILIES, rotation=25, ha="right")
    ax.set_ylim(0, 1.02)
    _panel(ax, "C")
    _style(ax, title=f"Saved-layer transfer @ depth {best_layer}", xlabel="train family", ylabel="accuracy", legend=True, legend_loc="lower right")

    ax = axes[1, 1]
    xs = np.arange(len(calibration_summary))
    fams = [r["family"] for r in calibration_summary]
    eces = [float(r.get("ece", 0.0)) for r in calibration_summary]
    briers = [float(r.get("brier", 0.0)) for r in calibration_summary]
    ax.bar(xs - 0.18, briers, 0.36, label="Brier", color="#56B4E9")
    ax.bar(xs + 0.18, eces, 0.36, label="ECE", color="#CC79A7")
    ax.set_xticks(xs)
    ax.set_xticklabels(fams, rotation=25, ha="right")
    _panel(ax, "D")
    _style(ax, title=f"Calibration costs @ logistic peak depth {truth_peak_layer}", xlabel="family", ylabel="lower is better", legend=True, legend_loc="upper left")
    fig.suptitle(f"Lab 4 probe evidence dashboard — surface feature: final word contains {surface_letter!r}", fontsize=14)
    bench.save_figure(ctx, fig, "probe_evidence_dashboard.png", "One-screen Lab 4 skepticism packet: accuracy, selectivity, transfer, and calibration.")


def plot_family_depth_atlas(ctx: bench.RunContext, report: list[dict[str, Any]], selectivity_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    max_depth = _depth_extent(report)
    depths = list(range(max_depth + 1))
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.2), constrained_layout=True, sharex=True, sharey=True)
    panels = [
        (axes[0, 0], "logistic accuracy", "accuracy", "logistic", None, 0.0, 1.0, "RdYlGn"),
        (axes[0, 1], "mass-mean accuracy", "accuracy", "mass_mean", None, 0.0, 1.0, "RdYlGn"),
        (axes[1, 0], "logistic selectivity", "selectivity", "logistic", selectivity_rows, -0.35, 0.65, "coolwarm"),
        (axes[1, 1], "mass-mean selectivity", "selectivity", "mass_mean", selectivity_rows, -0.35, 0.65, "coolwarm"),
    ]
    for ax, title, kind, method, sel_rows, vmin, vmax, cmap in panels:
        grid = np.full((len(FAMILIES), len(depths)), np.nan)
        for i, family in enumerate(FAMILIES):
            for j, layer in enumerate(depths):
                if kind == "accuracy":
                    v = _acc(report, method=method, eval_kind="within", layer=layer, train_family=family, eval_family=family)
                else:
                    v = _selectivity_lookup(selectivity_rows, family, method, layer)
                if v is not None:
                    grid[i, j] = v
        im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_yticks(range(len(FAMILIES)))
        ax.set_yticklabels(FAMILIES)
        ax.set_xticks([d for d in depths if d % max(1, len(depths) // 6) == 0 or d == depths[-1]])
        ax.set_xlabel("depth")
        fig.colorbar(im, ax=ax, fraction=0.046)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Family-depth atlas: the probe does not emerge everywhere at once")
    bench.save_figure(ctx, fig, "family_depth_atlas.png", "Heatmap atlas of within-family accuracy and selectivity across depth for both probe types.")


def plot_control_ladder(ctx: bench.RunContext, control_rows: list[dict[str, Any]], truth_peak_layer: int, surface_letter: str) -> None:
    fig, ax = bench.new_figure(figsize=(10.6, 5.6))
    y_positions = list(range(len(control_rows)))
    metrics = [
        ("majority_baseline", "majority", CONTROL_COLORS["majority"], "v"),
        ("token_length_baseline", "length", CONTROL_COLORS["length"], "P"),
        ("random_direction_accuracy", "random", CONTROL_COLORS["random"], "s"),
        ("shuffled_control_accuracy", "shuffled", CONTROL_COLORS["shuffled"], "x"),
        ("surface_accuracy", "surface", CONTROL_COLORS["surface"], "D"),
        ("truth_logistic_accuracy", "truth logistic", METHOD_COLORS["logistic"], "o"),
    ]
    ax.axvspan(0.45, 0.55, color="#BBBBBB", alpha=0.18)
    for y, row in zip(y_positions, control_rows):
        shuffled = _float_or_none(row.get("shuffled_control_accuracy"))
        real = _float_or_none(row.get("truth_logistic_accuracy"))
        if shuffled is not None and real is not None:
            ax.plot([shuffled, real], [y, y], color=_lighten(family_color(row["family"]), 0.55), linewidth=5, alpha=0.55, solid_capstyle="round")
        for key, label, color, marker in metrics:
            v = _float_or_none(row.get(key))
            if v is not None:
                scatter_kwargs = {"color": color, "marker": marker, "s": 70, "label": label if y == 0 else None, "zorder": 5}
                if marker not in {"x", "+", "1", "2", "3", "4"}:
                    scatter_kwargs.update({"edgecolor": "white", "linewidth": 0.55})
                ax.scatter(v, y, **scatter_kwargs)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["family"] for r in control_rows])
    ax.set_xlim(0, 1.02)
    ax.invert_yaxis()
    ax.set_xlabel("accuracy")
    ax.set_title(f"Control ladder at logistic peak depth {truth_peak_layer}\nsurface = final word contains {surface_letter!r}")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    bench.save_figure(ctx, fig, "probe_control_ladder.png", "Per-family truth accuracy compared to shuffled, random, length, majority, and surface controls.")


def plot_logistic_massmean_gap(ctx: bench.RunContext, report: list[dict[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(10.2, 5.4))
    max_depth = _depth_extent(report)
    for family in FAMILIES:
        xs, ys = [], []
        for layer in range(max_depth + 1):
            logi = _acc(report, method="logistic", eval_kind="within", layer=layer, train_family=family, eval_family=family)
            mass = _acc(report, method="mass_mean", eval_kind="within", layer=layer, train_family=family, eval_family=family)
            if logi is not None and mass is not None:
                xs.append(layer)
                ys.append(logi - mass)
        if xs:
            ax.plot(xs, ys, color=family_color(family), linewidth=2.0, label=family)
            _label_end(ax, xs, ys, family, family_color(family))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("residual-stream depth")
    ax.set_ylabel("logistic accuracy minus mass-mean accuracy")
    ax.set_title("Flexible separator vs saved direction: when logistics outrun geometry")
    ax.legend(fontsize=8, loc="best")
    bench.save_figure(ctx, fig, "logistic_vs_massmean_gap.png", "Difference between logistic and mass-mean truth accuracy by family and depth.")


def plot_truth_direction_projection_strip(
    ctx: bench.RunContext,
    X_layers: Any,
    labels: Any,
    families: list[str],
    best_layer: int,
    direction_family: str,
    final_probe: dict[str, Any],
    outlier_mask: Any | None = None,
) -> None:
    import numpy as np
    import torch

    direction = final_probe["direction"].detach().float()
    norm = direction.norm().clamp_min(1e-9)
    unit = direction / norm
    threshold = float(final_probe["threshold"] / float(norm))
    X = X_layers[:, best_layer, :]
    margin = (X @ unit).detach().float() - threshold
    fig, ax = bench.new_figure(figsize=(10.4, 5.7))
    rng = np.random.default_rng(0)
    for i, family in enumerate(FAMILIES):
        idxs = [j for j, f in enumerate(families) if f == family and (outlier_mask is None or not bool(outlier_mask[j]))]
        for label_value, name, offset, color in ((1, "true", -0.10, "#0072B2"), (0, "false", 0.10, "#D55E00")):
            vals = [float(margin[j]) for j in idxs if int(labels[j]) == label_value]
            if not vals:
                continue
            xs = i + offset + rng.normal(0, 0.018, size=len(vals))
            ax.scatter(xs, vals, color=color, s=28, alpha=0.72, edgecolor="white", linewidth=0.45, label=name if i == 0 else None)
            med = float(np.median(vals))
            ax.plot([i + offset - 0.08, i + offset + 0.08], [med, med], color=color, linewidth=2.4)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(range(len(FAMILIES)))
    ax.set_xticklabels(FAMILIES, rotation=20, ha="right")
    ax.set_ylabel("signed margin along saved unit direction\n(positive = predicts true)")
    ax.set_title(f"Saved truth direction @ depth {best_layer}, trained on {direction_family}\nA clean DECODE artifact separates rows; Lab 7 asks whether moving it changes behavior")
    ax.legend(fontsize=8, loc="best")
    bench.save_figure(ctx, fig, "truth_direction_projection_strip.png", "Signed margins along the saved mass-mean direction for true and false statements by family.")


def plot_split_balance(ctx: bench.RunContext, split_audit: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    counts = {(f, s, l): 0 for f in FAMILIES for s in ("train", "eval") for l in ("true", "false")}
    for r in split_audit:
        f = r["family"]
        s = r["split"]
        counts[(f, s, "true")] += int(r["n_true"])
        counts[(f, s, "false")] += int(r["n_false"])
    fig, ax = plt.subplots(figsize=(9.8, 4.8), constrained_layout=True)
    x = np.arange(len(FAMILIES))
    width = 0.34
    train_true = [counts[(f, "train", "true")] for f in FAMILIES]
    train_false = [counts[(f, "train", "false")] for f in FAMILIES]
    eval_true = [counts[(f, "eval", "true")] for f in FAMILIES]
    eval_false = [counts[(f, "eval", "false")] for f in FAMILIES]
    ax.bar(x - width / 2, train_true, width, label="train true", color="#0072B2")
    ax.bar(x - width / 2, train_false, width, bottom=train_true, label="train false", color="#D55E00")
    ax.bar(x + width / 2, eval_true, width, label="eval true", color=_lighten("#0072B2", 0.35), hatch="//")
    ax.bar(x + width / 2, eval_false, width, bottom=eval_true, label="eval false", color=_lighten("#D55E00", 0.35), hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(FAMILIES, rotation=20, ha="right")
    ax.set_ylabel("statement count")
    ax.set_title("Grouped split balance audit: no one-class train/eval trapdoors")
    ax.legend(fontsize=8, ncol=2)
    _style(ax, legend=False)
    bench.save_figure(ctx, fig, "split_balance_audit.png", "Train/eval true/false counts by family after grouped leakage-resistant splitting.")


def plot_norm_outlier_trajectories(ctx: bench.RunContext, row_norms: Any, statements: list[Statement], mid_depth: int) -> None:
    import numpy as np

    norms = row_norms.detach().float().numpy()
    depths = np.arange(norms.shape[1])
    median = np.median(norms, axis=0)
    scores = norms[:, mid_depth] / max(float(np.median(norms[:, mid_depth])), 1e-9)
    order = np.argsort(scores)[::-1][: min(10, len(scores))]
    fig, ax = bench.new_figure(figsize=(10.2, 5.6))
    ax.plot(depths, median, color="#222222", linewidth=2.3, label="median row")
    for rank, idx in enumerate(order):
        s = statements[int(idx)]
        color = family_color(s.family)
        ax.plot(depths, norms[idx], color=color, alpha=0.35 if rank > 4 else 0.75, linewidth=1.3 if rank > 4 else 1.8)
        if rank < 6:
            _label_end(ax, list(depths), list(norms[idx]), s.statement_id, color)
    if float(np.max(norms)) / max(float(np.median(norms)), 1e-9) > 12:
        ax.set_yscale("symlog", linthresh=max(float(np.median(norms)), 1.0))
    ax.set_xlabel("residual-stream depth")
    ax.set_ylabel("raw activation norm at final token")
    ax.set_title(f"Norm-outlier trajectories ranked at depth {mid_depth}: the rows that can bend a mean")
    ax.legend(fontsize=8, loc="upper left")
    bench.save_figure(ctx, fig, "norm_outlier_trajectories.png", "Top activation-norm rows over depth, showing which statements can dominate a mass-mean direction.")



def build_mass_mean_influence_rows(
    X_layers: Any,
    labels: Any,
    families: list[str],
    statements: list[Statement],
    train_mask: Any,
    row_norms: Any,
    best_layer: int,
) -> list[dict[str, Any]]:
    """Leave-one-training-row-out sensitivity of the saved-layer class-mean direction."""
    import torch

    rows: list[dict[str, Any]] = []
    median_norm = float(row_norms[:, best_layer].median())
    for family in FAMILIES:
        fam_idx = torch.tensor([i for i, f in enumerate(families) if f == family])
        train_idx = fam_idx[train_mask[fam_idx]]
        eval_idx = fam_idx[~train_mask[fam_idx]]
        if len(train_idx) < 4 or len(eval_idx) == 0:
            continue
        Xtr = X_layers[train_idx][:, best_layer, :]
        ytr = labels[train_idx]
        if len(set(ytr.tolist())) < 2:
            continue
        base = fit_mass_mean(Xtr, ytr)
        base_dir = base["direction"] / base["direction"].norm().clamp_min(1e-9)
        base_acc = eval_mass_mean(base, X_layers[eval_idx][:, best_layer, :], labels[eval_idx])
        for local_i, global_i in enumerate(train_idx.tolist()):
            keep = torch.ones(len(train_idx), dtype=torch.bool)
            keep[local_i] = False
            if len(set(ytr[keep].tolist())) < 2:
                continue
            probe = fit_mass_mean(Xtr[keep], ytr[keep])
            d = probe["direction"] / probe["direction"].norm().clamp_min(1e-9)
            cos = float(torch.clamp(base_dir @ d, -1.0, 1.0))
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            acc = eval_mass_mean(probe, X_layers[eval_idx][:, best_layer, :], labels[eval_idx])
            rows.append(
                {
                    "statement_id": statements[int(global_i)].statement_id,
                    "family": family,
                    "label": int(labels[int(global_i)]),
                    "layer": best_layer,
                    "raw_norm": round(float(row_norms[int(global_i), best_layer]), 4),
                    "raw_norm_ratio_to_median": round(float(row_norms[int(global_i), best_layer]) / max(median_norm, 1e-9), 4),
                    "cosine_without_row": round(cos, 6),
                    "angle_shift_degrees": round(angle, 4),
                    "baseline_eval_accuracy": round(base_acc, 4),
                    "eval_accuracy_without_row": round(acc, 4),
                    "eval_accuracy_delta_without_row": round(acc - base_acc, 4),
                    "influence_score": round(angle + 25.0 * abs(acc - base_acc), 4),
                    "statement": statements[int(global_i)].statement,
                }
            )
    rows.sort(key=lambda r: float(r["influence_score"]), reverse=True)
    return rows


def plot_mass_mean_influence(ctx: bench.RunContext, influence_rows: list[dict[str, Any]], best_layer: int) -> None:
    import matplotlib.pyplot as plt

    if not influence_rows:
        return
    top = influence_rows[: min(14, len(influence_rows))]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, max(5.4, 0.32 * len(top) + 2.0)), constrained_layout=True)
    ax = axes[0]
    for family in FAMILIES:
        rows = [r for r in influence_rows if r["family"] == family]
        if not rows:
            continue
        ax.scatter(
            [float(r["raw_norm_ratio_to_median"]) for r in rows],
            [float(r["angle_shift_degrees"]) for r in rows],
            s=[32 + 800 * abs(float(r["eval_accuracy_delta_without_row"])) for r in rows],
            color=family_color(family),
            marker=family_marker(family),
            alpha=0.66,
            edgecolor="white",
            linewidth=0.45,
            label=family,
        )
    for r in top[:7]:
        ax.annotate(r["statement_id"], (float(r["raw_norm_ratio_to_median"]), float(r["angle_shift_degrees"])), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("raw norm / median at saved layer")
    ax.set_ylabel("direction shift when row is removed (degrees)")
    ax.set_title(f"Leave-one-row-out mass-mean leverage @ depth {best_layer}\npoint size = eval accuracy change")
    ax.legend(fontsize=8)

    ax = axes[1]
    labels_plot = [f"{r['statement_id']}\n{r['family']}" for r in reversed(top)]
    vals = [float(r["influence_score"]) for r in reversed(top)]
    ax.barh(range(len(top)), vals, color=[family_color(r["family"]) for r in reversed(top)], alpha=0.82)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels_plot, fontsize=7)
    ax.set_xlabel("influence score = angle shift + 25×|eval acc delta|")
    ax.set_title("Largest leverage rows")
    fig.suptitle("Mean-direction leverage: raw-norm outliers are only one way a row can bend geometry")
    bench.save_figure(ctx, fig, "mass_mean_influence.png", "Leave-one-training-row-out sensitivity for mass-mean truth directions at the saved layer.")

def plot_probe_evidence_matrix(ctx: bench.RunContext, evidence_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    columns = [
        ("logistic\nacc", "logistic_accuracy_at_truth_peak", False),
        ("logistic\nselectivity", "logistic_selectivity_at_truth_peak", False),
        ("random\ncontrol", "random_direction_at_truth_peak", True),
        ("surface\nacc", "surface_accuracy_at_truth_peak", True),
        ("length\nbase", "token_length_baseline", True),
        ("mass-mean\nacc", "mass_mean_accuracy_at_saved_layer", False),
        ("min cross\ntransfer", "minimum_mass_mean_cross_family_at_saved_layer", False),
        ("misconceptions\ntransfer", "misconceptions_transfer_at_saved_layer", False),
        ("ECE", "calibration_ece_at_truth_peak", True),
    ]
    data = np.full((len(evidence_rows), len(columns)), np.nan)
    display = [["" for _ in columns] for _ in evidence_rows]
    for i, row in enumerate(evidence_rows):
        for j, (_, key, lower_better) in enumerate(columns):
            v = _float_or_none(row.get(key))
            if v is None:
                continue
            display[i][j] = f"{v:.2f}" if key != "logistic_selectivity_at_truth_peak" else f"{v:+.2f}"
            score = 1.0 - v if lower_better else v
            if key == "logistic_selectivity_at_truth_peak":
                score = 0.5 + 0.5 * max(-0.5, min(0.5, v)) / 0.5
            data[i, j] = max(0.0, min(1.0, score))
    fig, ax = plt.subplots(figsize=(11.5, 4.6), constrained_layout=True)
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([c[0] for c in columns], rotation=0)
    ax.set_yticks(range(len(evidence_rows)))
    ax.set_yticklabels([r["family"] for r in evidence_rows])
    for i in range(len(evidence_rows)):
        for j in range(len(columns)):
            if display[i][j]:
                ax.text(j, i, display[i][j], ha="center", va="center", fontsize=8.5)
    ax.set_title("Probe evidence matrix: every family carries its controls with it")
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, fraction=0.030, label="green = stronger evidence / cleaner control")
    bench.save_figure(ctx, fig, "probe_evidence_matrix.png", "Table-like heatmap of per-family probe accuracy, controls, transfer, and calibration.")


# ---------------------------------------------------------------------------
# Artifact cards and table helpers
# ---------------------------------------------------------------------------


def choose_best_layer(report: list[dict[str, Any]], n_depths: int) -> tuple[int, float]:
    def cross_and_within(layer: int) -> tuple[float, float]:
        # Negations and misconceptions are excluded from the primary layer
        # criterion. An affirmative-trained direction can anti-predict negations
        # or fail on popular false beliefs; those are stress-test results to
        # report, not quantities to optimize away.
        cross = [
            r["accuracy"]
            for r in report
            if r["method"] == "mass_mean"
            and r["layer"] == layer
            and r["eval_kind"] == "cross"
            and r["eval_family"] in AFFIRMATIVE_FAMILIES
            and r["train_family"] in AFFIRMATIVE_FAMILIES
        ]
        within = [
            r["accuracy"]
            for r in report
            if r["method"] == "mass_mean"
            and r["layer"] == layer
            and r["eval_kind"] == "within"
            and r["eval_family"] in AFFIRMATIVE_FAMILIES
        ]
        return (min(cross) if cross else 0.0, statistics.fmean(within) if within else 0.0)

    best_layer = max(range(n_depths), key=lambda k: cross_and_within(k))
    return best_layer, cross_and_within(best_layer)[0]


def peak(report: list[dict[str, Any]], track: str, method: str, eval_kind: str) -> tuple[int, float]:
    rows = [r for r in report if r["track"] == track and r["method"] == method and r["eval_kind"] == eval_kind]
    by_layer: dict[int, list[float]] = {}
    for r in rows:
        if r["layer"] >= 0:
            by_layer.setdefault(int(r["layer"]), []).append(float(r["accuracy"]))
    means = {k: statistics.fmean(v) for k, v in by_layer.items()}
    k = max(means, key=means.get)
    return k, means[k]


def mean_acc(report: list[dict[str, Any]], track: str, method: str, eval_kind: str, layer: int | None = None) -> float | None:
    vals = [
        r["accuracy"]
        for r in report
        if r["track"] == track
        and r["method"] == method
        and r["eval_kind"] == eval_kind
        and (layer is None or r["layer"] == layer)
    ]
    return statistics.fmean(vals) if vals else None


def build_selectivity_rows(report: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for method in ("logistic", "mass_mean"):
            real = {
                r["layer"]: r["accuracy"]
                for r in report
                if r["track"] == "truth"
                and r["method"] == method
                and r["eval_kind"] == "within"
                and r["train_family"] == family
            }
            ctrl = {
                r["layer"]: r["accuracy"]
                for r in report
                if r["track"] == "truth"
                and r["method"] == method
                and r["eval_kind"] == "shuffled_control"
                and r["train_family"] == family
            }
            rnd = {
                r["layer"]: r["accuracy"]
                for r in report
                if r["track"] == "truth"
                and r["method"] == "random_direction"
                and r["eval_kind"] == "random_control"
                and r["train_family"] == family
            }
            for layer in sorted(set(real) & set(ctrl)):
                rows.append(
                    {
                        "family": family,
                        "method": method,
                        "layer": layer,
                        "real_accuracy": real[layer],
                        "shuffled_accuracy": ctrl[layer],
                        "random_direction_accuracy": rnd.get(layer, ""),
                        "selectivity": round(real[layer] - ctrl[layer], 4),
                    }
                )
    return rows


def write_truth_direction_card(
    ctx: bench.RunContext,
    metadata: dict[str, Any],
    direction_transfer: dict[str, float],
    below_chance: list[str],
) -> None:
    transfer_lines = [f"- {fam}: {acc:.3f}" for fam, acc in sorted(direction_transfer.items())]
    below = ", ".join(below_chance) if below_chance else "none"
    norm_note = str(metadata.get("normalization", "unknown"))
    norm_card_line = str(metadata.get("normalization_card", norm_note))
    lines = [
        "# Lab 4 truth direction card",
        "",
        "## What this vector is",
        "",
        (
            "A mass-mean direction, true-class mean minus false-class mean, trained on "
            f"{norm_card_line} from `{metadata['train_family']}` at stream depth {metadata['layer']}."
        ),
        "It is a DECODE artifact. It is not evidence that the model uses this direction.",
        "",
        "## Convention",
        "",
        f"- model: `{metadata['model_id']}`",
        f"- stream depth: {metadata['layer']} with bench `streams[k]` convention",
        "- position: final token of the frozen statement",
        f"- normalization: {norm_note}",
        "- saved tensor: `tables/truth_direction.pt`",
        "- readable metadata: `tables/truth_direction_metadata.json`",
        "",
        "## Transfer accuracies",
        "",
        *transfer_lines,
        "",
        f"Below-chance transfer families: {below}",
        "",
        "## How Lab 7 should use it",
        "",
        "Lab 7 may test whether injecting this direction changes True/False assent. A positive result",
        "would be a new CAUSAL claim. A negative result would mean decodable-but-inert under that",
        "intervention, not that the Lab 4 probe was invalid.",
        "",
    ]
    card_path = ctx.path("tables", "truth_direction_card.md")
    bench.write_text(card_path, "\n".join(lines))
    ctx.register_artifact(card_path, "card", "Human-readable metadata and caveats for the saved truth direction.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    import torch

    args = ctx.args
    if args.prompt_set not in ("small", "medium", "full"):
        raise ValueError(
            "Lab 4 uses frozen truth CSVs in data/. --prompt-set selects size only: "
            "small, medium, or full."
        )

    # Load and cap families. --max-examples is a PER-FAMILY cap here.
    per_family_cap = args.max_examples if args.max_examples > 0 else 0
    if args.prompt_set == "small" and not per_family_cap:
        per_family_cap = 20
    elif args.prompt_set == "medium" and not per_family_cap:
        per_family_cap = 40

    family_counts: list[dict[str, Any]] = []
    statements: list[Statement] = []
    for family in FAMILIES:
        fam_raw = load_family(family)
        fam = cap_balanced(fam_raw, per_family_cap)
        statements.extend(fam)
        n_true = sum(s.label for s in fam)
        family_counts.append(
            {
                "family": family,
                "data_file": DATA_FILES[family],
                "sha256": data_file_digest(family),
                "raw_n": len(fam_raw),
                "used_n": len(fam),
                "used_true": n_true,
                "used_false": len(fam) - n_true,
            }
        )
        print(f"[lab4] {family}: {len(fam)} statements ({n_true} true, {len(fam) - n_true} false)")

    data_manifest_path = ctx.path("diagnostics", "frozen_data_manifest.json")
    normalization_key = "row_unit_norm" if NORMALIZE_ROWS else "raw_streams"
    normalization_note = (
        "each activation row unit-normalized before fitting; apply the direction to unit-normalized streams "
        "or rescale by local stream norm"
        if NORMALIZE_ROWS
        else "raw final-token residual streams; no row-wise normalization was applied before fitting"
    )
    normalization_card = (
        "unit-normalized final-token residual streams"
        if NORMALIZE_ROWS
        else "raw final-token residual streams"
    )
    bench.write_json(
        data_manifest_path,
        {
            "families": family_counts,
            "normalization": normalization_key,
            "normalization_note": normalization_note,
        },
    )
    ctx.register_artifact(data_manifest_path, "diagnostic", "Frozen truth CSV counts and hashes.")

    split_lookup, split_audit = make_grouped_split(statements)
    split_path = ctx.path("diagnostics", "split_audit.csv")
    bench.write_csv_with_context(ctx, split_path, split_audit)
    ctx.register_artifact(split_path, "diagnostic", "Grouped train/eval split audit, including leakage-resistant group keys.")

    surface_letter = pick_surface_letter(statements, split_lookup)
    print(f"[lab4] surface-track feature: final word contains {surface_letter!r}")

    # Instrument verification. Lab 4 is not about hook plumbing, but it still
    # dies if the stream convention is wrong.
    bench.run_hook_parity_check(ctx, bundle, statements[0].statement)
    first_capture = bench.run_with_residual_cache(bundle, statements[0].statement)
    bench.run_lens_self_check(ctx, bundle, first_capture)

    # One forward pass per statement. Keep the final-position stream stack.
    n_depths = bundle.anatomy.n_layers + 1
    feats = []
    n_tokens_list = []
    final_token_texts = []
    t0_report = max(1, len(statements) // 5)
    for i, s in enumerate(statements):
        capture = first_capture if i == 0 else bench.run_with_residual_cache(bundle, s.statement)
        feats.append(capture.streams[:, -1, :])      # [L+1, d] fp32 cpu
        n_tokens_list.append(len(capture.input_ids))
        final_token_texts.append(bundle.tokenizer.decode([int(capture.input_ids[-1])]))
        if (i + 1) % t0_report == 0:
            print(f"[lab4] cached {i + 1}/{len(statements)} statements")

    X_layers_raw = torch.stack(feats)                 # [n, L+1, d]
    row_norms = X_layers_raw.norm(dim=-1)             # [n, L+1]
    X_layers = X_layers_raw / row_norms[..., None].clamp_min(1e-9) if NORMALIZE_ROWS else X_layers_raw
    labels = torch.tensor([s.label for s in statements])
    families = [s.family for s in statements]
    surface_labels = torch.tensor([1 if surface_letter in final_word(s.statement).lower() else 0 for s in statements])
    n_tokens = torch.tensor(n_tokens_list, dtype=torch.float32)

    mid = n_depths // 2
    median_mid_norm = float(row_norms[:, mid].median())
    manifest = []
    for i, s in enumerate(statements):
        manifest.append(
            {
                "statement_id": s.statement_id,
                "family": s.family,
                "statement": s.statement,
                "label": s.label,
                "split_key": split_group_key(s),
                "split": "train" if split_lookup[s.statement_id] else "eval",
                "surface_label": int(surface_labels[i]),
                "n_tokens": int(n_tokens[i]),
                "final_token_text": final_token_texts[i],
                "stream_norm_mid": round(float(row_norms[i, mid]), 3),
                "stream_norm_final": round(float(row_norms[i, -1]), 3),
                "norm_outlier": bool(row_norms[i, mid] > OUTLIER_MULTIPLIER * median_mid_norm),
            }
        )
    outliers = [m["statement_id"] for m in manifest if m["norm_outlier"]]
    if outliers:
        print(f"[lab4] activation-norm outliers (>{OUTLIER_MULTIPLIER:.1f}x median at depth {mid}): {outliers}")
    man_path = ctx.path("tables", "statement_manifest.csv")
    bench.write_csv_with_context(ctx, man_path, manifest)
    ctx.register_artifact(man_path, "table", "Every statement with split, labels, token count, final token, and norm diagnostics.")

    fam_idx = {f: [i for i, ff in enumerate(families) if ff == f] for f in FAMILIES}
    train_mask = torch.tensor([split_lookup[s.statement_id] for s in statements])

    # ----- probe sweep ------------------------------------------------------
    report: list[dict[str, Any]] = []
    logistic_probes: dict[tuple[int, str], dict[str, Any]] = {}
    mass_mean_probes: dict[tuple[int, str], dict[str, Any]] = {}

    def add(
        track: str,
        method: str,
        layer: int,
        train_family: str,
        eval_family: str,
        eval_kind: str,
        acc: float,
        n_train: int,
        n_eval: int,
        *,
        brier: float | None = None,
        nll: float | None = None,
        ece: float | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "track": track,
            "method": method,
            "layer": layer,
            "train_family": train_family,
            "eval_family": eval_family,
            "eval_kind": eval_kind,
            "accuracy": round(float(acc), 4),
            "n_train": n_train,
            "n_eval": n_eval,
        }
        if brier is not None:
            row["brier"] = round(float(brier), 4)
        if nll is not None:
            row["nll"] = round(float(nll), 4)
        if ece is not None:
            row["ece"] = round(float(ece), 4)
        report.append(row)

    print(f"[lab4] probing {n_depths} depths x {len(FAMILIES)} families x 2 methods (+controls)")
    for layer in range(n_depths):
        X = X_layers[:, layer, :]
        for family in FAMILIES:
            idx = torch.tensor(fam_idx[family])
            tr = idx[train_mask[idx]]
            ev = idx[~train_mask[idx]]
            others = {f: torch.tensor(fam_idx[f]) for f in FAMILIES if f != family}

            for method, fit, evaluate in (
                ("logistic", fit_logistic, eval_logistic),
                ("mass_mean", fit_mass_mean, eval_mass_mean),
            ):
                probe = fit(X[tr], labels[tr])
                if method == "logistic":
                    logistic_probes[(layer, family)] = probe
                    metrics = eval_logistic_metrics(probe, X[ev], labels[ev])
                    add(
                        "truth",
                        method,
                        layer,
                        family,
                        family,
                        "within",
                        metrics["accuracy"],
                        len(tr),
                        len(ev),
                        brier=metrics["brier"],
                        nll=metrics["nll"],
                        ece=metrics["ece"],
                    )
                else:
                    mass_mean_probes[(layer, family)] = probe
                    add("truth", method, layer, family, family, "within", evaluate(probe, X[ev], labels[ev]), len(tr), len(ev))

                for of, oidx in others.items():
                    add("truth", method, layer, family, of, "cross", evaluate(probe, X[oidx], labels[oidx]), len(tr), len(oidx))

                ctrl_accs = []
                for shuffle_i in range(N_SHUFFLES):
                    y_shuf = shuffled_labels(
                        labels[tr],
                        seed=args.seed * 1009 + layer * 31 + FAMILIES.index(family) * 131 + shuffle_i,
                    )
                    ctrl = fit(X[tr], y_shuf)
                    ctrl_accs.append(evaluate(ctrl, X[ev], labels[ev]))
                add("truth", method, layer, family, family, "shuffled_control", statistics.fmean(ctrl_accs), len(tr), len(ev))

            add(
                "truth",
                "random_direction",
                layer,
                family,
                family,
                "random_control",
                eval_random_directions(X[tr], labels[tr], X[ev], labels[ev], seed=args.seed * 7919 + layer * 101 + FAMILIES.index(family)),
                len(tr),
                len(ev),
            )

            if labels_have_two_classes(surface_labels[tr]):
                sprobe = fit_logistic(X[tr], surface_labels[tr])
                smetrics = eval_logistic_metrics(sprobe, X[ev], surface_labels[ev])
                add(
                    "surface",
                    "logistic",
                    layer,
                    family,
                    family,
                    "within",
                    smetrics["accuracy"],
                    len(tr),
                    len(ev),
                    brier=smetrics["brier"],
                    nll=smetrics["nll"],
                    ece=smetrics["ece"],
                )
            else:
                add(
                    "surface",
                    "majority_baseline",
                    layer,
                    family,
                    family,
                    "surface_untrainable_control",
                    majority_baseline(surface_labels[tr], surface_labels[ev]),
                    len(tr),
                    len(ev),
                )

    # Layer-independent controls.
    for family in FAMILIES:
        idx = torch.tensor(fam_idx[family])
        tr = idx[train_mask[idx]]
        ev = idx[~train_mask[idx]]
        lp = fit_logistic(n_tokens[tr][:, None], labels[tr])
        lmetrics = eval_logistic_metrics(lp, n_tokens[ev][:, None], labels[ev])
        add(
            "truth",
            "token_length_baseline",
            -1,
            family,
            family,
            "length_control",
            lmetrics["accuracy"],
            len(tr),
            len(ev),
            brier=lmetrics["brier"],
            nll=lmetrics["nll"],
            ece=lmetrics["ece"],
        )
        add("truth", "majority_baseline", -1, family, family, "majority_control", majority_baseline(labels[tr], labels[ev]), len(tr), len(ev))

    report_path = ctx.path("tables", "probe_report.csv")
    bench.write_csv_with_context(ctx, report_path, report)
    ctx.register_artifact(report_path, "table", "Every probe evaluation: track, method, depth, families, controls, and calibration metrics where defined.")
    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, report)
    ctx.register_artifact(results_path, "results", "Alias of probe_report.csv for the standard run contract.")

    selectivity_rows = build_selectivity_rows(report)
    selectivity_path = ctx.path("tables", "selectivity_report.csv")
    bench.write_csv_with_context(ctx, selectivity_path, selectivity_rows)
    ctx.register_artifact(selectivity_path, "table", "Real-vs-shuffled selectivity by depth, family, and probe type.")

    # ----- best layer and saved truth direction -----------------------------
    best_layer, best_min_cross = choose_best_layer(report, n_depths)
    print(f"[lab4] best affirmative cross-family depth: {best_layer} (worst transfer acc {best_min_cross:.3f})")

    def worst_transfer(train_family: str) -> float:
        vals = [
            r["accuracy"]
            for r in report
            if r["method"] == "mass_mean"
            and r["layer"] == best_layer
            and r["train_family"] == train_family
            and r["eval_kind"] == "cross"
            and r["eval_family"] in SAVED_DIRECTION_EVAL_FAMILIES
        ]
        return min(vals) if vals else 0.0

    direction_family = max(AFFIRMATIVE_FAMILIES, key=worst_transfer)
    fam_t = torch.tensor(fam_idx[direction_family])
    tr = fam_t[train_mask[fam_t]]
    final_probe = fit_mass_mean(X_layers[tr][:, best_layer, :], labels[tr])
    print(
        f"[lab4] saved direction: {direction_family}-trained mass-mean @ depth {best_layer} "
        f"(worst transfer {worst_transfer(direction_family):.3f})"
    )

    direction_rows = [
        r
        for r in report
        if r["method"] == "mass_mean"
        and r["layer"] == best_layer
        and r["train_family"] == direction_family
        and r["eval_kind"] in ("within", "cross")
    ]
    direction_transfer = {r["eval_family"]: r["accuracy"] for r in direction_rows}
    direction_worst_cross = min((r["accuracy"] for r in direction_rows if r["eval_kind"] == "cross"), default=None)
    direction_selection_worst_cross = worst_transfer(direction_family)
    below_chance = [f for f, a in sorted(direction_transfer.items()) if f != direction_family and a < 0.5]

    direction_metadata = {
        "direction_key": f"{direction_family}_mass_mean_depth_{best_layer}",
        "method": "mass_mean (difference of class means)",
        "train_family": f"{direction_family} (train split)",
        "layer": best_layer,
        "layer_convention": "depth index into bench streams[k]: 0 = embeddings, k = residual after block k, k = n_layers is final-norm input",
        "position": "final token of the statement",
        "stream": "pre-norm residual, bench streams[k] convention",
        "normalization": normalization_note,
        "normalization_key": normalization_key,
        "normalization_card": normalization_card,
        "model_id": bundle.anatomy.model_id,
        "d_model": bundle.anatomy.d_model,
        "n_layers": bundle.anatomy.n_layers,
        "train_statement_ids": [statements[int(i)].statement_id for i in tr.tolist()],
        "metrics": {
            "within": next((r["accuracy"] for r in direction_rows if r["eval_kind"] == "within"), None),
            **{f"cross_{ef}": direction_transfer.get(ef) for ef in FAMILIES if ef != direction_family},
            "selection_worst_cross": direction_selection_worst_cross,
            "worst_cross_including_stress_tests": direction_worst_cross,
        },
    }

    direction_path = ctx.path("tables", "truth_direction.pt")
    torch.save(
        {
            "direction": final_probe["direction"],
            "threshold": final_probe["threshold"],
            **direction_metadata,
        },
        direction_path,
    )
    ctx.register_artifact(direction_path, "tensor", "Mass-mean truth direction at the best cross-family depth; Lab 7 reuses this causally.")
    metadata_path = ctx.path("tables", "truth_direction_metadata.json")
    bench.write_json(metadata_path, direction_metadata)
    ctx.register_artifact(metadata_path, "metadata", "Readable metadata for truth_direction.pt.")
    write_truth_direction_card(ctx, direction_metadata, direction_transfer, below_chance)

    # ----- aggregate metrics and calibration artifacts ----------------------
    truth_peak_layer, truth_peak_acc = peak(report, "truth", "logistic", "within")
    mass_peak_layer, mass_peak_acc = peak(report, "truth", "mass_mean", "within")
    surface_peak_layer, surface_peak_acc = peak(report, "surface", "logistic", "within")
    truth_peak_ctrl = mean_acc(report, "truth", "logistic", "shuffled_control", truth_peak_layer)
    mass_peak_ctrl = mean_acc(report, "truth", "mass_mean", "shuffled_control", mass_peak_layer)
    length_accs = [r["accuracy"] for r in report if r["eval_kind"] == "length_control"]
    majority_accs = [r["accuracy"] for r in report if r["eval_kind"] == "majority_control"]

    calibration_rows: list[dict[str, Any]] = []
    calibration_summary: list[dict[str, Any]] = []
    for family in FAMILIES:
        idx = torch.tensor(fam_idx[family])
        ev = idx[~train_mask[idx]]
        probe = logistic_probes[(truth_peak_layer, family)]
        probs = logistic_probs(probe, X_layers[ev][:, truth_peak_layer, :]).detach().float()
        calibration_rows.extend(calibration_bins(probs, labels[ev], family=family, layer=truth_peak_layer))
        metrics = eval_logistic_metrics(probe, X_layers[ev][:, truth_peak_layer, :], labels[ev])
        calibration_summary.append({"family": family, "layer": truth_peak_layer, **{k: round(v, 4) for k, v in metrics.items()}})
    calibration_path = ctx.path("tables", "calibration_curve.csv")
    bench.write_csv_with_context(ctx, calibration_path, calibration_rows)
    ctx.register_artifact(calibration_path, "table", "Reliability-curve bins for the logistic truth probe at its peak depth.")
    calibration_summary_path = ctx.path("tables", "calibration_summary.csv")
    bench.write_csv_with_context(ctx, calibration_summary_path, calibration_summary)
    ctx.register_artifact(calibration_summary_path, "table", "Accuracy, Brier score, NLL, and ECE for peak-depth logistic truth probes.")

    evidence_rows = build_probe_evidence_rows(report, selectivity_rows, calibration_summary, truth_peak_layer, best_layer)
    evidence_path = ctx.path("tables", "probe_evidence_matrix.csv")
    bench.write_csv_with_context(ctx, evidence_path, evidence_rows)
    ctx.register_artifact(evidence_path, "table", "Per-family evidence matrix: accuracy, controls, selectivity, transfer, surface baseline, and calibration.")

    control_ladder_rows = build_control_ladder_rows(report, truth_peak_layer, surface_letter)
    control_ladder_path = ctx.path("tables", "control_ladder_summary.csv")
    bench.write_csv_with_context(ctx, control_ladder_path, control_ladder_rows)
    ctx.register_artifact(control_ladder_path, "table", "Per-family peak-depth truth accuracy beside shuffled, random, surface, length, and majority controls.")

    influence_rows = build_mass_mean_influence_rows(X_layers, labels, families, statements, train_mask, row_norms, best_layer)
    influence_path = ctx.path("tables", "mass_mean_influence_report.csv")
    bench.write_csv_with_context(ctx, influence_path, influence_rows)
    ctx.register_artifact(influence_path, "table", "Leave-one-training-row-out sensitivity of mass-mean directions at the saved layer.")

    plot_guide_rows = lab4_plot_reading_guide()
    plot_guide_path = ctx.path("tables", "plot_reading_guide.csv")
    bench.write_csv_with_context(ctx, plot_guide_path, plot_guide_rows)
    ctx.register_artifact(plot_guide_path, "table", "Reading guide that maps each Lab 4 plot to the concept it teaches.")

    metrics = {
        "n_statements": len(statements),
        "per_family_cap": per_family_cap,
        "surface_letter": surface_letter,
        "normalization": normalization_key,
        "activation_norm_outliers": outliers,
        "truth_peak": {"layer": truth_peak_layer, "accuracy": truth_peak_acc},
        "truth_peak_shuffled_control": truth_peak_ctrl,
        "truth_peak_selectivity": truth_peak_acc - truth_peak_ctrl if truth_peak_ctrl is not None else None,
        "mass_mean_peak": {"layer": mass_peak_layer, "accuracy": mass_peak_acc},
        "mass_mean_peak_shuffled_control": mass_peak_ctrl,
        "surface_peak": {"layer": surface_peak_layer, "accuracy": surface_peak_acc},
        "best_cross_family_layer": best_layer,
        "best_min_cross_accuracy": best_min_cross,
        "saved_direction": {
            "train_family": direction_family,
            "layer": best_layer,
            "worst_cross_accuracy": direction_selection_worst_cross,
            "worst_cross_including_stress_tests": direction_worst_cross,
            "transfer_accuracies": direction_transfer,
            "below_chance_transfer_families": below_chance,
        },
        "token_length_baseline_mean": statistics.fmean(length_accs) if length_accs else None,
        "majority_baseline_mean": statistics.fmean(majority_accs) if majority_accs else None,
        "peak_logistic_calibration": calibration_summary,
        "top_mass_mean_influence_rows": influence_rows[:5],
    }
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 4 metrics.")

    # ----- plots ------------------------------------------------------------
    if not args.no_plots:
        outlier_mask = row_norms[:, mid] > OUTLIER_MULTIPLIER * median_mid_norm
        plot_probe_dashboard(ctx, report, selectivity_rows, calibration_summary, surface_letter, best_layer, truth_peak_layer)
        plot_decodability(ctx, report, surface_letter, best_layer, truth_peak_layer)
        plot_family_depth_atlas(ctx, report, selectivity_rows)
        plot_generalization(ctx, report, best_layer)
        plot_selectivity(ctx, selectivity_rows)
        plot_control_ladder(ctx, control_ladder_rows, truth_peak_layer, surface_letter)
        plot_logistic_massmean_gap(ctx, report)
        plot_truth_direction_projection_strip(ctx, X_layers, labels, families, best_layer, direction_family, final_probe, outlier_mask)
        plot_projection_panels(ctx, X_layers, labels, families, n_depths, outlier_mask)
        plot_norm_diagnostics(ctx, row_norms, n_depths)
        plot_norm_outlier_trajectories(ctx, row_norms, statements, mid)
        plot_mass_mean_influence(ctx, influence_rows, best_layer)
        plot_calibration(ctx, calibration_rows, truth_peak_layer)
        plot_split_balance(ctx, split_audit)
        plot_probe_evidence_matrix(ctx, evidence_rows)

    # ----- cards, claims, summary ------------------------------------------
    run_name = ctx.run_dir.name
    truth_ctrl_text = f"{truth_peak_ctrl:.2f}" if truth_peak_ctrl is not None else "n/a"
    truth_selectivity_text = f"{truth_peak_acc - truth_peak_ctrl:+.2f}" if truth_peak_ctrl is not None else "n/a"
    length_text = f"{metrics['token_length_baseline_mean']:.2f}" if metrics["token_length_baseline_mean"] is not None else "n/a"
    majority_text = f"{metrics['majority_baseline_mean']:.2f}" if metrics["majority_baseline_mean"] is not None else "n/a"
    direction_worst_text = f"{direction_selection_worst_cross:.3f}"
    direction_stress_worst_text = f"{direction_worst_cross:.3f}" if direction_worst_cross is not None else "n/a"
    transfer_text = ", ".join(
        f"{family} {direction_transfer[family]:.2f}" + (" (within)" if family == direction_family else "")
        for family in FAMILIES
        if family in direction_transfer
    )

    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "DECODE",
            "text": (
                f"Truth is linearly decodable from {bundle.anatomy.model_id}'s residual stream: "
                f"within-family held-out accuracy peaks at {truth_peak_acc:.2f} (depth {truth_peak_layer}, "
                f"logistic, mean over {len(FAMILIES)} families), versus a same-depth shuffled-label "
                f"control of {truth_ctrl_text} (selectivity {truth_selectivity_text}), a token-length "
                f"baseline of {length_text}, and a majority baseline of {majority_text}."
            ),
            "artifact": f"runs/{run_name}/tables/probe_report.csv",
            "falsifier": "A matched syntactic family with no truth content shows the same accuracy and selectivity.",
        },
        {
            "id": f"{LAB_ID}-C2",
            "tag": "DECODE",
            "text": (
                f"The saved {direction_family}-trained mass-mean direction at depth {best_layer} transfers "
                f"across the frozen families ({transfer_text})"
                + (
                    f"; on {', '.join(below_chance)} it is below chance, which is anti-correlation rather "
                    "than absence of structure"
                    if below_chance
                    else ""
                )
                + "; this is the vector written to truth_direction.pt for Lab 7's causal test."
            ),
            "artifact": f"runs/{run_name}/tables/truth_direction_card.md",
            "falsifier": "A new held-out family such as dates or physical facts drops transfer to chance.",
        },
        {
            "id": f"{LAB_ID}-C3",
            "tag": "DECODE",
            "text": (
                f"Decodability alone is cheap: the surface feature (final word contains {surface_letter!r}) "
                f"is decodable at {surface_peak_acc:.2f} (peak depth {surface_peak_layer}). Any claim built "
                "on 'a probe found it' must explain why the found information is not surface-grade."
            ),
            "artifact": f"runs/{run_name}/plots/decodability_by_layer.png",
            "falsifier": "N/A, this is the lab's calibration claim rather than a mechanistic truth claim.",
        },
    ]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)

    # A compact deliverable card, separate from the standard run summary.
    claim_card_path = ctx.path("probe_claim_card.md")
    bench.write_text(
        claim_card_path,
        "\n".join(
            [
                "# Lab 4 probe claim card",
                "",
                "## Verdict",
                "",
                f"Truth is selectively decodable at peak logistic depth {truth_peak_layer}, but this is still DECODE evidence.",
                f"The saved mass-mean direction lives at depth {best_layer} and is trained on {direction_family}.",
                "",
                "## Controls checked",
                "",
                f"- shuffled-label control at peak depth: {truth_ctrl_text}",
                f"- token-length baseline mean: {length_text}",
                f"- majority baseline mean: {majority_text}",
                f"- random-direction controls: {N_RANDOM_DIRS} per depth",
                "- grouped train/eval split audit: diagnostics/split_audit.csv",
                "- misconceptions stress test: plots/generalization_matrix.png",
                "- activation norm diagnostics: tables/statement_manifest.csv and plots/activation_norms_by_depth.png",
                "- calibration: tables/calibration_summary.csv and plots/truth_calibration_curve.png",
                "- probe evidence matrix: tables/probe_evidence_matrix.csv and plots/probe_evidence_matrix.png",
                "- control ladder: tables/control_ladder_summary.csv and plots/probe_control_ladder.png",
                "- saved-vector projection strip: plots/truth_direction_projection_strip.png",
                "- mass-mean leverage: tables/mass_mean_influence_report.csv and plots/mass_mean_influence.png",
                "",
                "## Non-claim",
                "",
                "This run does not show that the model uses the truth direction, believes the statements, or would answer correctly.",
                "Lab 7 is the first causal test of the saved vector.",
                "",
            ]
        ),
    )
    ctx.register_artifact(claim_card_path, "card", "One-page DECODE claim card with controls, caveats, and non-claims.")

    lines = [
        "# Lab 4 run summary: probing with controls",
        "",
        "## Run identity",
        "",
        f"- model: `{bundle.anatomy.model_id}` ({bundle.anatomy.n_layers} blocks, d_model {bundle.anatomy.d_model})",
        f"- statements: {len(statements)} across {len(FAMILIES)} frozen families (per-family cap {per_family_cap or 'none'})",
        f"- probe position: final token | probes: logistic (LBFGS, L2={LOGISTIC_L2}) + mass-mean",
        f"- split: grouped by leakage-resistant keys at train fraction {TRAIN_FRACTION}",
        "- evidence level: `DECODE` — nothing here shows the model USES these directions (that test is Lab 7)",
        "",
        "## 1. What behavior was studied?",
        "",
        "No generation behavior was studied directly. The lab probes REPRESENTATIONS (pre-final-norm residual at",
        "statement end, same convention as Labs 1-3): truth of frozen statements, plus a deliberately shallow",
        "final-word feature as the calibration track that shows 'decodability is cheap'.",
        "",
        "## 2. What internal object was measured?",
        "",
        "Linear decodability from the final-token pre-norm residual stream at every depth, with logistic and",
        "mass-mean probes. The saved object is a mass-mean truth direction for Lab 7's causal test.",
        "",
        "## 3. What controls were used?",
        "",
        f"Grouped train/eval split audit, shuffled-label refits (x{N_SHUFFLES}), random directions (x{N_RANDOM_DIRS}),",
        "token-length and majority baselines, family-held-out transfer including negations, surface-track calibration,",
        "activation-norm diagnostics (the outlier specimen that can hijack a mean), and logistic calibration.",
        "",
        "## 4. Headline numbers",
        "",
        f"- truth peak (logistic, within-family): {truth_peak_acc:.3f} at depth {truth_peak_layer}/{n_depths - 1}",
        f"- same-depth shuffled-label control: {truth_ctrl_text} (selectivity {truth_selectivity_text})",
        f"- surface peak: {surface_peak_acc:.3f} at depth {surface_peak_layer}",
        f"- token-length baseline: {metrics['token_length_baseline_mean']:.3f}",
        f"- majority baseline: {metrics['majority_baseline_mean']:.3f}",
        f"- best affirmative cross-family depth (mass-mean, worst transfer): {best_layer} ({best_min_cross:.3f})",
        f"- saved direction: `tables/truth_direction.pt` ({direction_family} mass-mean @ depth {best_layer}/{n_depths - 1}, selection worst transfer {direction_worst_text}; stress-test worst transfer {direction_stress_worst_text})",
        "",
        "## 5. What claim is supported, and at what evidence level?",
        "",
    ]
    for c in claims:
        lines.append(f"- `{c['id']}` {c['tag']}: {c['text']}")
        lines.append(f"  - falsifier: {c['falsifier']}")
    lines += [
        "",
        "## 6. The reading order",
        "",
        "1. `probe_claim_card.md` for the verdict and non-claims.",
        "2. `diagnostics/split_audit.csv` and `tables/statement_manifest.csv` before trusting any accuracy.",
        "3. `plots/probe_evidence_dashboard.png` for the whole skepticism packet on one screen.",
        "4. `plots/decodability_by_layer.png` and `plots/family_depth_atlas.png` for headline curves plus family-specific emergence.",
        "5. `plots/generalization_matrix.png` for family transfer, negation inversions, and misconception stress-test results.",
        "6. `plots/probe_control_ladder.png`, `plots/selectivity_by_layer.png`, and `tables/selectivity_report.csv` for real-minus-control evidence.",
        "7. `plots/truth_direction_projection_strip.png` and `tables/truth_direction_card.md` before using the vector in Lab 7.",
        "8. `plots/truth_calibration_curve.png` and `tables/calibration_summary.csv` for confidence hygiene.",
        "9. `plots/norm_outlier_trajectories.png`, `plots/mass_mean_influence.png`, and `plots/split_balance_audit.png` for row-norm, leverage, and split-hygiene audits.",
        "",
        "## 7. Caveats students must carry forward",
        "",
        "- DECODE is not USE. The saved direction's causal test (does adding it actually change",
        "  behavior toward truth on held-out items?) is Lab 7's job. This lab only earns the",
        "  'accessible linear information on these families' part of any claim.",
        "- Truth here means truth on four frozen families with the specific templates and",
        "  grouped splits used. The misconception family is a stress test, not a saved-direction selector.",
        "- Below-chance negation transfer can be structured anti-correlation (the probe read",
        "  the surface polarity), not mere failure. This is often the most informative result.",
        "- Logistic and mass-mean probes license different claims. Disagreement is data, not",
        "  noise. The surface track on the same activations shows that even a trivial feature",
        "  can look 'deep' if you only look at raw accuracy without controls.",
        "",
    ]
    summary_path = ctx.path("run_summary.md")
    bench.write_text(summary_path, "\n".join(lines))
    ctx.register_artifact(summary_path, "summary", "The seven standard questions answered with this run's numbers.")
    print(f"[lab4] wrote run_summary.md and {len(claims)} drafted ledger claims")
