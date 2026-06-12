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


FAMILY_COLORS = {"cities": "tab:blue", "comparisons": "tab:orange", "negations": "tab:green",
                 "misconceptions": "tab:red"}


def family_color(family: str) -> str:
    """Color for a statement family; unknown families get a stable fallback
    instead of a KeyError (adding a CSV family must never crash a plot)."""
    return FAMILY_COLORS.get(family, "tab:gray")


def _mean_curve(report: list[dict[str, Any]], predicate) -> tuple[list[int], list[float]]:
    by_layer: dict[int, list[float]] = {}
    for r in report:
        if predicate(r) and r["layer"] >= 0:
            by_layer.setdefault(int(r["layer"]), []).append(float(r["accuracy"]))
    depths = sorted(by_layer)
    return depths, [statistics.fmean(by_layer[k]) for k in depths]


def plot_decodability(
    ctx: bench.RunContext,
    report: list[dict[str, Any]],
    surface_letter: str,
    best_layer: int,
    truth_peak_layer: int,
) -> None:
    """Headline figure: truth, surface, and controls on one set of axes."""
    fig, ax = bench.new_figure(figsize=(10.2, 6.0))

    curves = [
        (
            "truth, logistic within-family",
            lambda r: r["track"] == "truth" and r["method"] == "logistic" and r["eval_kind"] == "within",
            {"linewidth": 2.6, "color": "tab:red"},
        ),
        (
            "truth, mass-mean within-family",
            lambda r: r["track"] == "truth" and r["method"] == "mass_mean" and r["eval_kind"] == "within",
            {"linewidth": 2.6, "color": "tab:purple", "linestyle": "--"},
        ),
        (
            f"surface, final word contains {surface_letter!r}",
            lambda r: r["track"] == "surface" and r["method"] == "logistic" and r["eval_kind"] == "within",
            {"linewidth": 2.0, "color": "tab:gray"},
        ),
        (
            "truth shuffled-label control",
            lambda r: r["track"] == "truth" and r["method"] == "logistic" and r["eval_kind"] == "shuffled_control",
            {"linewidth": 1.6, "color": "black", "linestyle": ":"},
        ),
        (
            "truth random-direction control",
            lambda r: r["track"] == "truth" and r["method"] == "random_direction",
            {"linewidth": 1.6, "color": "tab:brown", "linestyle": ":"},
        ),
    ]
    for label, pred, style in curves:
        depths, vals = _mean_curve(report, pred)
        if depths:
            ax.plot(depths, vals, label=label, **style)

    ax.axvline(best_layer, color="tab:purple", linewidth=1.0, alpha=0.45,
               label=f"saved direction depth {best_layer}")
    if truth_peak_layer != best_layer:
        ax.axvline(truth_peak_layer, color="tab:red", linewidth=0.8, alpha=0.3,
                   label=f"logistic peak depth {truth_peak_layer}")
    ax.axhline(0.5, color="black", linewidth=0.7, alpha=0.6)
    ax.fill_between([0, max([best_layer, truth_peak_layer, 1])], 0.45, 0.55, alpha=0.08, color="black",
                    label="chance neighborhood")
    ax.set_xlabel("residual-stream depth (0 = embeddings, k = after k blocks)")
    ax.set_ylabel("held-out accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Probe accuracy by depth, with controls riding the same rails")
    ax.legend(fontsize=8, loc="lower right")
    bench.save_figure(
        ctx,
        fig,
        "decodability_by_layer.png",
        "Surface vs truth decodability by depth, with shuffled and random controls.",
    )


def plot_generalization(ctx: bench.RunContext, report: list[dict[str, Any]], best_layer: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    im = None
    for panel_i, (ax, method) in enumerate(zip(axes, ("logistic", "mass_mean"))):
        grid = np.full((len(FAMILIES), len(FAMILIES)), np.nan)
        for r in report:
            if (
                r["track"] == "truth"
                and r["method"] == method
                and r["layer"] == best_layer
                and r["eval_kind"] in ("within", "cross")
            ):
                i = FAMILIES.index(r["train_family"])
                j = FAMILIES.index(r["eval_family"])
                grid[i, j] = r["accuracy"]
        im = ax.imshow(grid, cmap="RdYlGn", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(FAMILIES)))
        ax.set_xticklabels(FAMILIES)
        ax.set_yticks(range(len(FAMILIES)))
        ax.set_yticklabels(FAMILIES)
        ax.set_xlabel("evaluated on")
        ax.set_ylabel("trained on" if panel_i == 0 else "")
        ax.set_title(f"{method} @ depth {best_layer}")
        for i in range(len(FAMILIES)):
            for j in range(len(FAMILIES)):
                if not np.isnan(grid[i, j]):
                    weight = "bold" if grid[i, j] < 0.5 else "normal"
                    ax.annotate(f"{grid[i, j]:.2f}", (j, i), ha="center", va="center",
                                fontsize=10, fontweight=weight)
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.03, label="accuracy")
    fig.suptitle("Family-held-out transfer, where below chance can be a real anti-feature")
    bench.save_figure(
        ctx,
        fig,
        "generalization_matrix.png",
        "Train-family x eval-family accuracy for both probe types at the saved direction depth.",
    )


def plot_selectivity(ctx: bench.RunContext, selectivity_rows: list[dict[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(9.3, 5.0))
    for family in FAMILIES:
        rows = [r for r in selectivity_rows if r["family"] == family and r["method"] == "logistic"]
        rows = sorted(rows, key=lambda r: r["layer"])
        if rows:
            ax.plot(
                [r["layer"] for r in rows],
                [r["selectivity"] for r in rows],
                linewidth=2.0,
                color=family_color(family),
                label=family,
            )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("residual-stream depth")
    ax.set_ylabel("selectivity (real accuracy minus shuffled-control accuracy)")
    ax.set_title("Selectivity by family: accuracy after subtracting shuffled-label controls")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "selectivity_by_layer.png", "Per-family truth-probe selectivity by depth.")


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
    fig, axes = plt.subplots(1, len(depths), figsize=(3.55 * len(depths), 3.8), sharey=False)
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
        ax.scatter(x_proj[yc == 1], y_proj[yc == 1], s=24, color="tab:green", alpha=0.82, label="true")
        ax.scatter(x_proj[yc == 0], y_proj[yc == 0], s=24, color="tab:red", alpha=0.82, label="false")
        ax.set_title(f"depth {k}", fontsize=10)
        ax.set_xlabel("mass-mean direction")
        ax.grid(True, alpha=0.28)
    axes[0].set_ylabel("top orthogonal PC")
    axes[0].legend(fontsize=8)
    fig.suptitle("Cities: truth separation over depth (raw-norm outliers hidden from view only)")
    fig.tight_layout()
    bench.save_figure(
        ctx,
        fig,
        "truth_projection_panels.png",
        "2-D projections of city statements at five depths.",
    )


def plot_norm_diagnostics(ctx: bench.RunContext, row_norms: Any, n_depths: int) -> None:
    import numpy as np

    norms = row_norms.numpy()
    depths = list(range(n_depths))
    med = np.median(norms, axis=0)
    p95 = np.quantile(norms, 0.95, axis=0)
    mx = np.max(norms, axis=0)
    fig, ax = bench.new_figure(figsize=(9.2, 5.0))
    ax.plot(depths, med, linewidth=2.0, label="median")
    ax.plot(depths, p95, linewidth=1.8, linestyle="--", label="95th percentile")
    ax.plot(depths, mx, linewidth=1.8, linestyle=":", label="max")
    ax.set_xlabel("residual-stream depth")
    ax.set_ylabel("raw activation norm at final token")
    ax.set_title("Activation-norm diagnostics: one row can bend a mass-mean direction")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "activation_norms_by_depth.png", "Median, p95, and max final-token stream norms by depth.")


def plot_calibration(ctx: bench.RunContext, calibration_rows: list[dict[str, Any]], layer: int) -> None:
    fig, ax = bench.new_figure(figsize=(6.2, 5.4))
    for family in FAMILIES:
        rows = [r for r in calibration_rows if r["family"] == family and r["n"]]
        if not rows:
            continue
        ax.plot(
            [float(r["mean_predicted_prob"]) for r in rows],
            [float(r["empirical_true_rate"]) for r in rows],
            marker="o",
            linewidth=1.8,
            color=family_color(family),
            label=family,
        )
    ax.plot([0, 1], [0, 1], linestyle=":", color="black", linewidth=1.0, label="perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted P(true)")
    ax.set_ylabel("empirical true rate")
    ax.set_title(f"Logistic truth-probe calibration at depth {layer}")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "truth_calibration_curve.png", "Calibration curve for the logistic truth probe at its peak depth.")


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
    }
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 4 metrics.")

    # ----- plots ------------------------------------------------------------
    if not args.no_plots:
        plot_decodability(ctx, report, surface_letter, best_layer, truth_peak_layer)
        plot_generalization(ctx, report, best_layer)
        plot_selectivity(ctx, selectivity_rows)
        outlier_mask = row_norms[:, mid] > OUTLIER_MULTIPLIER * median_mid_norm
        plot_projection_panels(ctx, X_layers, labels, families, n_depths, outlier_mask)
        plot_norm_diagnostics(ctx, row_norms, n_depths)
        plot_calibration(ctx, calibration_rows, truth_peak_layer)

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
        "3. `plots/decodability_by_layer.png` for the headline curves and controls.",
        "4. `plots/generalization_matrix.png` for family transfer, negation inversions, and misconception stress-test results.",
        "5. `plots/selectivity_by_layer.png` and `tables/selectivity_report.csv` for real-minus-control evidence.",
        "6. `plots/truth_calibration_curve.png` and `tables/calibration_summary.csv` for confidence hygiene.",
        "7. `tables/truth_direction_card.md` before using the vector in Lab 7.",
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
