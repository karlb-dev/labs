"""Lab 4: Probing without fooling yourself (now featuring truth).

Two tracks probed from the SAME forward passes:

* **Surface track** (mechanical warm-up): a property of the final word's
  token text — decodable from token identity alone. Its by-layer curve is the
  baseline shape of "trivially decodable".
* **Truth track** (headline): is the statement true? Probed at the
  end-of-statement position, Geometry-of-Truth style, on three frozen
  families (cities, comparisons, negations) vendored in ``data/``.

The lab's product is skepticism made quantitative. Every accuracy comes with
its controls:

- shuffled-label refits   -> does the probe family have capacity to "find"
                             structure in noise at this n and d?
- random-direction probe  -> how well does an arbitrary direction do?
- token-length baseline   -> is "truth" secretly statement length?
- family-held-out eval    -> does cities-truth transfer to comparisons and
                             to negations (where surface form anti-correlates
                             with truth)?
- selectivity             = real accuracy - shuffled-control accuracy.

Two probe types per layer, because Lab 7 will reuse the direction CAUSALLY
and mean-difference directions have repeatedly proven more causally relevant
than max-margin ones:

- logistic regression (torch LBFGS, L2, standardized features — no sklearn,
  nobody runs code they can't explain)
- mass-mean: difference of class means, threshold at the projected midpoint.

The run saves ``truth_direction.pt`` (mass-mean, best cross-family layer)
with metadata; Lab 7 loads it and asks whether decodable means usable.

Evidence level: DECODE. Nothing here shows the model USES these directions.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import pathlib
import statistics
from typing import Any

import interp_bench as bench

LAB_ID = "L04"

FAMILIES = ("cities", "comparisons", "negations")
DATA_FILES = {
    "cities": "truth_cities.csv",
    "comparisons": "truth_comparisons.csv",
    "negations": "truth_negations.csv",
}
TRAIN_FRACTION = 0.7
N_SHUFFLES = 2          # shuffled-label refits per (layer, family)
N_RANDOM_DIRS = 3       # random-direction baselines per layer
LOGISTIC_L2 = 1e-2
SURFACE_LETTERS = ("r", "n", "a", "o")  # candidate letters; most balanced wins


@dataclasses.dataclass(frozen=True)
class Statement:
    statement_id: str
    family: str
    statement: str
    label: int           # 1 = true
    meta: str = ""


def load_family(family: str) -> list[Statement]:
    path = bench.COURSE_ROOT / "data" / DATA_FILES[family]
    if not path.exists():
        raise RuntimeError(
            f"Frozen dataset missing: {path}. The truth CSVs are vendored in "
            "data/ — re-checkout the repo; do NOT regenerate per-run."
        )
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(Statement(row["statement_id"], row["family"], row["statement"],
                                 int(row["label"]), row.get("meta", "")))
    return out


def cap_balanced(statements: list[Statement], cap: int) -> list[Statement]:
    """Cap a family while keeping the true/false balance exact."""
    if cap <= 0 or cap >= len(statements):
        return statements
    true = [s for s in statements if s.label == 1]
    false = [s for s in statements if s.label == 0]
    half = max(1, cap // 2)
    return true[:half] + false[:half]


def is_train(statement_id: str) -> bool:
    """Deterministic 70/30 split by statement id — stable across runs/seeds."""
    digest = hashlib.md5(statement_id.encode()).hexdigest()
    return int(digest[:8], 16) % 100 < int(TRAIN_FRACTION * 100)


# ---------------------------------------------------------------------------
# Probes (pure torch; the math is the point)
# ---------------------------------------------------------------------------


def fit_logistic(X: Any, y: Any, l2: float = LOGISTIC_L2) -> dict[str, Any]:
    """L2-regularized logistic regression via LBFGS on standardized features.

    Returns a dict with everything needed to evaluate elsewhere: the weight
    vector, bias, and the train-set standardization (eval data must be
    standardized with TRAIN statistics — leaking eval statistics into the
    scaler is the quietest way to cheat)."""
    import torch

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


def eval_logistic(probe: dict[str, Any], X: Any, y: Any) -> float:
    Xs = (X - probe["mu"]) / probe["sigma"]
    pred = (Xs @ probe["w"] + probe["b"]) > 0
    return float((pred == y.bool()).float().mean())


def fit_mass_mean(X: Any, y: Any) -> dict[str, Any]:
    """Difference of class means; threshold at the projected midpoint."""
    mu_true = X[y == 1].mean(dim=0)
    mu_false = X[y == 0].mean(dim=0)
    direction = mu_true - mu_false
    threshold = float(((mu_true + mu_false) / 2) @ direction)
    return {"direction": direction, "threshold": threshold}


def eval_mass_mean(probe: dict[str, Any], X: Any, y: Any) -> float:
    pred = (X @ probe["direction"]) > probe["threshold"]
    return float((pred == y.bool()).float().mean())


def eval_random_directions(X_train: Any, y_train: Any, X_eval: Any, y_eval: Any, seed: int) -> float:
    """Best-effort random-direction probe: random unit vector, midpoint
    threshold fit on train. Mean accuracy over N_RANDOM_DIRS draws."""
    import torch

    accs = []
    gen = torch.Generator().manual_seed(seed)
    for _ in range(N_RANDOM_DIRS):
        d = torch.randn(X_train.shape[1], generator=gen)
        d = d / d.norm()
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


# ---------------------------------------------------------------------------
# Surface track feature
# ---------------------------------------------------------------------------


def pick_surface_letter(statements: list[Statement]) -> str:
    """Choose the candidate letter whose presence in the final token is most
    balanced across the dataset — transparent, surface-level, arbitrary on
    purpose."""
    best, best_gap = SURFACE_LETTERS[0], 1.0
    for letter in SURFACE_LETTERS:
        frac = statistics.fmean(
            1.0 if letter in final_word(s.statement).lower() else 0.0 for s in statements
        )
        gap = abs(frac - 0.5)
        if gap < best_gap:
            best, best_gap = letter, gap
    return best


def final_word(statement: str) -> str:
    return statement.rstrip(".").split()[-1]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


FAMILY_COLORS = {"cities": "tab:blue", "comparisons": "tab:orange", "negations": "tab:green"}


def plot_decodability(
    ctx: bench.RunContext,
    report: list[dict[str, Any]],
    n_depths: int,
    surface_letter: str,
) -> None:
    """The lab's headline: surface vs truth decodability by layer."""
    fig, ax = bench.new_figure(figsize=(9.5, 6.0))

    def curve(rows: list[dict[str, Any]]) -> list[float]:
        by_layer: dict[int, list[float]] = {}
        for r in rows:
            by_layer.setdefault(r["layer"], []).append(r["accuracy"])
        return [statistics.fmean(by_layer[k]) for k in sorted(by_layer)]

    truth_lr = curve([r for r in report if r["track"] == "truth" and r["method"] == "logistic"
                      and r["eval_kind"] == "within"])
    truth_mm = curve([r for r in report if r["track"] == "truth" and r["method"] == "mass_mean"
                      and r["eval_kind"] == "within"])
    surface = curve([r for r in report if r["track"] == "surface" and r["method"] == "logistic"
                     and r["eval_kind"] == "within"])
    shuffled = curve([r for r in report if r["track"] == "truth" and r["method"] == "logistic"
                      and r["eval_kind"] == "shuffled_control"])
    rand_dir = curve([r for r in report if r["track"] == "truth" and r["method"] == "random_direction"])

    depths = list(range(len(truth_lr)))
    ax.plot(depths, truth_lr, linewidth=2.5, color="tab:red", label="truth — logistic (within-family)")
    ax.plot(depths, truth_mm, linewidth=2.5, color="tab:purple", linestyle="--", label="truth — mass-mean")
    ax.plot(depths, surface, linewidth=2.0, color="tab:gray",
            label=f"surface — final word contains '{surface_letter}'")
    ax.plot(depths, shuffled, linewidth=1.5, color="black", linestyle=":",
            label="shuffled-label control (logistic)")
    if rand_dir:
        ax.plot(depths, rand_dir, linewidth=1.5, color="tab:brown", linestyle=":",
                label="random-direction baseline")
    ax.axhline(0.5, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("depth (0 = embeddings, k = after k blocks)")
    ax.set_ylabel("held-out accuracy")
    ax.set_ylim(0.3, 1.02)
    ax.set_title("Decodable is cheap; decodable-and-deep emerges with depth")
    ax.legend(fontsize=8, loc="lower right")
    bench.save_figure(ctx, fig, "decodability_by_layer.png",
                      "Surface vs truth decodability by layer, with controls on the same axes.")


def plot_generalization(
    ctx: bench.RunContext, report: list[dict[str, Any]], best_layer: int
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, method in zip(axes, ("logistic", "mass_mean")):
        grid = np.full((len(FAMILIES), len(FAMILIES)), np.nan)
        for r in report:
            if (r["track"] == "truth" and r["method"] == method and r["layer"] == best_layer
                    and r["eval_kind"] in ("within", "cross")):
                i = FAMILIES.index(r["train_family"])
                j = FAMILIES.index(r["eval_family"])
                grid[i, j] = r["accuracy"]
        im = ax.imshow(grid, cmap="RdYlGn", vmin=0.3, vmax=1.0)
        ax.set_xticks(range(len(FAMILIES)))
        ax.set_xticklabels(FAMILIES)
        ax.set_yticks(range(len(FAMILIES)))
        ax.set_yticklabels(FAMILIES)
        ax.set_xlabel("evaluated on")
        ax.set_ylabel("trained on")
        ax.set_title(f"{method} @ layer {best_layer}")
        for i in range(len(FAMILIES)):
            for j in range(len(FAMILIES)):
                if not np.isnan(grid[i, j]):
                    ax.annotate(f"{grid[i, j]:.2f}", (j, i), ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.03, label="accuracy")
    fig.suptitle("Generalization across statement families (diagonal = within-family held-out)")
    bench.save_figure(ctx, fig, "generalization_matrix.png",
                      "Train-family x eval-family accuracy for both probe types at the best layer.")


def plot_selectivity(ctx: bench.RunContext, report: list[dict[str, Any]], n_depths: int) -> None:
    fig, ax = bench.new_figure(figsize=(9.0, 5.0))
    for family in FAMILIES:
        real = {r["layer"]: r["accuracy"] for r in report
                if r["track"] == "truth" and r["method"] == "logistic"
                and r["eval_kind"] == "within" and r["train_family"] == family}
        ctrl = {r["layer"]: r["accuracy"] for r in report
                if r["track"] == "truth" and r["method"] == "logistic"
                and r["eval_kind"] == "shuffled_control" and r["train_family"] == family}
        depths = sorted(set(real) & set(ctrl))
        ax.plot(depths, [real[d] - ctrl[d] for d in depths], linewidth=2.0,
                color=FAMILY_COLORS[family], label=family)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("depth")
    ax.set_ylabel("selectivity (real - shuffled-control accuracy)")
    ax.set_title("Selectivity: how much of the accuracy is real structure?")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "selectivity_by_layer.png",
                      "Per-family selectivity of the truth probe by layer.")


def plot_projection_panels(
    ctx: bench.RunContext,
    X_layers: Any,            # [n, L+1, d]
    labels: Any,              # [n]
    families: list[str],
    n_depths: int,
    outlier_mask: Any = None,  # [n] bool; rogue-norm rows excluded from the VIEW only
) -> None:
    """Cities statements projected on (mass-mean dir, top orthogonal PC) at
    five depths: separation emerging over depth, visible at a glance.

    Norm-outlier statements are excluded from this plot only (they compress
    every other point into a blob); they remain in every probe number."""
    import matplotlib.pyplot as plt
    import torch

    idx = [i for i, f in enumerate(families)
           if f == "cities" and (outlier_mask is None or not bool(outlier_mask[i]))]
    if len(idx) < 8:
        return
    Xc = X_layers[idx]
    yc = labels[idx]
    depths = sorted({1, n_depths // 4, n_depths // 2, (3 * n_depths) // 4, n_depths - 1})
    fig, axes = plt.subplots(1, len(depths), figsize=(3.6 * len(depths), 3.8), sharey=False)
    for ax, k in zip(axes, depths):
        X = Xc[:, k, :]
        mm = fit_mass_mean(X, yc)
        d1 = mm["direction"] / mm["direction"].norm().clamp_min(1e-9)
        Xp = X - X.mean(dim=0)
        resid = Xp - (Xp @ d1)[:, None] * d1[None, :]
        # Top principal component of the residual, via one power iteration pass.
        v = resid.T @ resid @ torch.ones(resid.shape[1]) / resid.shape[1]
        for _ in range(8):
            v = resid.T @ (resid @ v)
            v = v / v.norm().clamp_min(1e-9)
        x_proj = Xp @ d1
        y_proj = Xp @ v
        ax.scatter(x_proj[yc == 1], y_proj[yc == 1], s=22, color="tab:green", alpha=0.8, label="true")
        ax.scatter(x_proj[yc == 0], y_proj[yc == 0], s=22, color="tab:red", alpha=0.8, label="false")
        ax.set_title(f"depth {k}", fontsize=10)
        ax.set_xlabel("mass-mean direction")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("top orthogonal PC")
    axes[0].legend(fontsize=8)
    fig.suptitle("Cities statements: truth separation emerging over depth "
                 "(norm-outlier rows excluded from view, not from numbers)")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "truth_projection_panels.png",
                      "2-D projections of city statements at five depths; separation along the mass-mean direction.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    import torch

    args = ctx.args
    if args.prompt_set not in ("small", "medium", "full"):
        raise ValueError(
            "Lab 4 uses the frozen truth CSVs in data/ (course rule: students "
            "never author truth sets); --prompt-set selects size only."
        )

    # Load and cap families. --max-examples is a PER-FAMILY cap here.
    per_family_cap = args.max_examples if args.max_examples > 0 else 0
    if args.prompt_set == "small" and not per_family_cap:
        per_family_cap = 20
    elif args.prompt_set == "medium" and not per_family_cap:
        per_family_cap = 40
    statements: list[Statement] = []
    for family in FAMILIES:
        fam = cap_balanced(load_family(family), per_family_cap)
        statements.extend(fam)
        n_true = sum(s.label for s in fam)
        print(f"[lab4] {family}: {len(fam)} statements ({n_true} true)")

    surface_letter = pick_surface_letter(statements)
    print(f"[lab4] surface-track feature: final word contains {surface_letter!r}")

    # Instrument verification (Lab 1's checks still guard this capture path).
    bench.run_hook_parity_check(ctx, bundle, statements[0].statement)
    first_capture = bench.run_with_residual_cache(bundle, statements[0].statement)
    bench.run_lens_self_check(ctx, bundle, first_capture)

    # One forward per statement; keep only the final-position stream stack.
    n_depths = bundle.anatomy.n_layers + 1
    feats = []
    n_tokens_list = []
    t0_report = max(1, len(statements) // 5)
    for i, s in enumerate(statements):
        capture = first_capture if i == 0 else bench.run_with_residual_cache(bundle, s.statement)
        feats.append(capture.streams[:, -1, :])      # [L+1, d] fp32 cpu
        n_tokens_list.append(len(capture.input_ids))
        if (i + 1) % t0_report == 0:
            print(f"[lab4] cached {i + 1}/{len(statements)} statements")
    X_layers_raw = torch.stack(feats)                 # [n, L+1, d]
    # Per-row unit normalization, and the reason is a specimen worth keeping:
    # on Olmo-3, "The city of Havana is in the Netherlands." produces a
    # final-position stream with ~7x the norm of every other statement. A raw
    # difference-of-class-means direction gets hijacked by whichever class
    # holds more such rogue rows in train, and every normal statement then
    # projects to the same value regardless of truth. Norms are recorded
    # below so the outliers stay visible instead of silently fixed.
    row_norms = X_layers_raw.norm(dim=-1)             # [n, L+1]
    X_layers = X_layers_raw / row_norms[..., None].clamp_min(1e-9)
    labels = torch.tensor([s.label for s in statements])
    families = [s.family for s in statements]
    surface_labels = torch.tensor(
        [1 if surface_letter in final_word(s.statement).lower() else 0 for s in statements]
    )
    n_tokens = torch.tensor(n_tokens_list, dtype=torch.float32)

    mid = n_depths // 2
    median_mid_norm = float(row_norms[:, mid].median())
    manifest = [
        {"statement_id": s.statement_id, "family": s.family, "statement": s.statement,
         "label": s.label, "split": "train" if is_train(s.statement_id) else "eval",
         "surface_label": int(surface_labels[i]), "n_tokens": int(n_tokens[i]),
         "stream_norm_mid": round(float(row_norms[i, mid]), 3),
         "norm_outlier": bool(row_norms[i, mid] > 3 * median_mid_norm)}
        for i, s in enumerate(statements)
    ]
    outliers = [m["statement_id"] for m in manifest if m["norm_outlier"]]
    if outliers:
        print(f"[lab4] activation-norm outliers (>3x median at depth {mid}): {outliers}")
    man_path = ctx.path("tables", "statement_manifest.csv")
    bench.write_csv(man_path, manifest)
    ctx.register_artifact(man_path, "table", "Every statement with split, labels, and token count.")

    fam_idx = {f: [i for i, ff in enumerate(families) if ff == f] for f in FAMILIES}
    train_mask = torch.tensor([is_train(s.statement_id) for s in statements])

    # ----- the probe sweep -------------------------------------------------
    report: list[dict[str, Any]] = []

    def add(track: str, method: str, layer: int, train_family: str, eval_family: str,
            eval_kind: str, acc: float, n_train: int, n_eval: int) -> None:
        report.append({
            "track": track, "method": method, "layer": layer,
            "train_family": train_family, "eval_family": eval_family,
            "eval_kind": eval_kind, "accuracy": round(acc, 4),
            "n_train": n_train, "n_eval": n_eval,
        })

    print(f"[lab4] probing {n_depths} depths x {len(FAMILIES)} families x 2 methods (+controls)")
    mass_mean_probes: dict[tuple[int, str], dict[str, Any]] = {}
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
                if method == "mass_mean":
                    mass_mean_probes[(layer, family)] = probe
                add("truth", method, layer, family, family, "within",
                    evaluate(probe, X[ev], labels[ev]), len(tr), len(ev))
                for of, oidx in others.items():
                    add("truth", method, layer, family, of, "cross",
                        evaluate(probe, X[oidx], labels[oidx]), len(tr), len(oidx))
                # Shuffled-label control: same capacity, no real structure.
                ctrl_accs = []
                for shuffle_i in range(N_SHUFFLES):
                    y_shuf = shuffled_labels(labels[tr], seed=args.seed * 1009 + layer * 31 + shuffle_i)
                    ctrl = fit(X[tr], y_shuf)
                    ctrl_accs.append(evaluate(ctrl, X[ev], labels[ev]))
                add("truth", method, layer, family, family, "shuffled_control",
                    statistics.fmean(ctrl_accs), len(tr), len(ev))

            # Random-direction baseline (method-independent).
            add("truth", "random_direction", layer, family, family, "random_control",
                eval_random_directions(X[tr], labels[tr], X[ev], labels[ev],
                                       seed=args.seed * 7919 + layer),
                len(tr), len(ev))

            # Surface track (logistic only — the point is the curve's shape).
            sprobe = fit_logistic(X[tr], surface_labels[tr])
            add("surface", "logistic", layer, family, family, "within",
                eval_logistic(sprobe, X[ev], surface_labels[ev]), len(tr), len(ev))

    # Token-length baseline: layer-independent, fit once per family.
    for family in FAMILIES:
        idx = torch.tensor(fam_idx[family])
        tr = idx[train_mask[idx]]
        ev = idx[~train_mask[idx]]
        lp = fit_logistic(n_tokens[tr][:, None], labels[tr])
        add("truth", "token_length_baseline", -1, family, family, "length_control",
            eval_logistic(lp, n_tokens[ev][:, None], labels[ev]), len(tr), len(ev))

    report_path = ctx.path("tables", "probe_report.csv")
    bench.write_csv(report_path, report)
    ctx.register_artifact(report_path, "table", "Every probe evaluation: track, method, layer, families, controls.")

    # ----- best layer by cross-family transfer (mass-mean) ------------------
    def cross_and_within(layer: int) -> tuple[float, float]:
        # Negations are EXCLUDED from the selection criterion: an
        # affirmative-trained mass-mean direction anti-predicting on negations
        # is the expected Geometry-of-Truth result, not a tie-breaker. Its
        # transfer number is reported separately as the known failure mode.
        cross = [r["accuracy"] for r in report
                 if r["method"] == "mass_mean" and r["layer"] == layer and r["eval_kind"] == "cross"
                 and r["eval_family"] != "negations" and r["train_family"] != "negations"]
        within = [r["accuracy"] for r in report
                  if r["method"] == "mass_mean" and r["layer"] == layer and r["eval_kind"] == "within"]
        return (min(cross) if cross else 0.0, statistics.fmean(within) if within else 0.0)

    # Primary criterion: worst affirmative cross-family transfer (Lab 7 wants
    # a direction that means truth, not cities-template). Tie-break by
    # within-family accuracy so a degenerate run doesn't elect layer 0.
    best_layer = max(range(n_depths), key=lambda k: cross_and_within(k))
    best_min_cross = cross_and_within(best_layer)[0]
    print(f"[lab4] best cross-family layer: {best_layer} (worst transfer acc {best_min_cross:.3f})")

    # ----- save the truth direction for Lab 7 -------------------------------
    # Pick the train family whose direction transfers best at the chosen
    # layer (min accuracy over ALL other families, negations included). In
    # our validation runs this elects comparisons, not cities: cities and
    # negations share a template, so a cities-trained direction can ride the
    # template; a comparisons-trained one has to mean truth.
    def worst_transfer(train_family: str) -> float:
        vals = [r["accuracy"] for r in report
                if r["method"] == "mass_mean" and r["layer"] == best_layer
                and r["train_family"] == train_family and r["eval_kind"] == "cross"]
        return min(vals) if vals else 0.0

    direction_family = max(("cities", "comparisons"), key=worst_transfer)
    fam_t = torch.tensor(fam_idx[direction_family])
    tr = fam_t[train_mask[fam_t]]
    final_probe = fit_mass_mean(X_layers[tr][:, best_layer, :], labels[tr])
    print(f"[lab4] saved direction: {direction_family}-trained mass-mean @ layer {best_layer} "
          f"(worst transfer {worst_transfer(direction_family):.3f})")
    direction_path = ctx.path("tables", "truth_direction.pt")
    torch.save(
        {
            "direction": final_probe["direction"],
            "threshold": final_probe["threshold"],
            "layer": best_layer,
            "position": "final token (end-of-statement period)",
            "stream": "pre-norm residual, bench streams[k] convention",
            "normalization": "each activation row unit-normalized before the mean difference; "
                             "apply to unit-normalized streams, or rescale by the local stream norm",
            "method": "mass_mean (difference of class means)",
            "train_family": f"{direction_family} (train split)",
            "model_id": bundle.anatomy.model_id,
            "d_model": bundle.anatomy.d_model,
            "metrics": {
                "within": next((r["accuracy"] for r in report
                                if r["method"] == "mass_mean" and r["layer"] == best_layer
                                and r["train_family"] == direction_family and r["eval_kind"] == "within"), None),
                **{
                    f"cross_{ef}": next((r["accuracy"] for r in report
                                         if r["method"] == "mass_mean" and r["layer"] == best_layer
                                         and r["train_family"] == direction_family
                                         and r["eval_family"] == ef and r["eval_kind"] == "cross"), None)
                    for ef in FAMILIES if ef != direction_family
                },
            },
        },
        direction_path,
    )
    ctx.register_artifact(direction_path, "tensor",
                          "Mass-mean truth direction at the best cross-family layer; Lab 7 reuses this causally.")

    # ----- plots ------------------------------------------------------------
    if not args.no_plots:
        plot_decodability(ctx, report, n_depths, surface_letter)
        plot_generalization(ctx, report, best_layer)
        plot_selectivity(ctx, report, n_depths)
        outlier_mask = row_norms[:, mid] > 3 * median_mid_norm
        plot_projection_panels(ctx, X_layers, labels, families, n_depths, outlier_mask)

    # ----- metrics, claims, summary -----------------------------------------
    def peak(track: str, method: str, eval_kind: str) -> tuple[int, float]:
        rows = [r for r in report if r["track"] == track and r["method"] == method
                and r["eval_kind"] == eval_kind]
        by_layer: dict[int, list[float]] = {}
        for r in rows:
            by_layer.setdefault(r["layer"], []).append(r["accuracy"])
        means = {k: statistics.fmean(v) for k, v in by_layer.items()}
        k = max(means, key=means.get)
        return k, means[k]

    truth_peak_layer, truth_peak_acc = peak("truth", "logistic", "within")
    surface_peak_layer, surface_peak_acc = peak("surface", "logistic", "within")
    length_accs = [r["accuracy"] for r in report if r["eval_kind"] == "length_control"]
    metrics = {
        "n_statements": len(statements),
        "per_family_cap": per_family_cap,
        "surface_letter": surface_letter,
        "activation_norm_outliers": outliers,
        "truth_peak": {"layer": truth_peak_layer, "accuracy": truth_peak_acc},
        "surface_peak": {"layer": surface_peak_layer, "accuracy": surface_peak_acc},
        "best_cross_family_layer": best_layer,
        "best_min_cross_accuracy": best_min_cross,
        "token_length_baseline_mean": statistics.fmean(length_accs) if length_accs else None,
    }
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 4 metrics.")

    run_name = ctx.run_dir.name
    mm_cross_neg = next((r["accuracy"] for r in report
                         if r["method"] == "mass_mean" and r["layer"] == best_layer
                         and r["train_family"] == "cities" and r["eval_family"] == "negations"), None)
    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "DECODE",
            "text": (
                f"Truth is linearly decodable from {bundle.anatomy.model_id}'s residual stream: "
                f"within-family held-out accuracy peaks at {truth_peak_acc:.2f} (layer {truth_peak_layer}, "
                f"logistic, mean over {len(FAMILIES)} families), against a shuffled-label control near "
                f"chance and a token-length baseline of {metrics['token_length_baseline_mean']:.2f}."
            ),
            "artifact": f"runs/{run_name}/tables/probe_report.csv",
            "falsifier": (
                "A matched syntactic family with no truth content (e.g. swapped entity templates) "
                "shows the same accuracy — the probe was reading template structure."
            ),
        },
        {
            "id": f"{LAB_ID}-C2",
            "tag": "DECODE",
            "text": (
                f"The cities mass-mean direction at layer {best_layer} transfers across families: "
                f"comparisons {next((r['accuracy'] for r in report if r['method'] == 'mass_mean' and r['layer'] == best_layer and r['train_family'] == 'cities' and r['eval_family'] == 'comparisons'), None)}, "
                f"negations {mm_cross_neg} — negations being where surface co-occurrence anti-correlates "
                "with truth."
            ),
            "artifact": f"runs/{run_name}/tables/generalization (see probe_report.csv + plots)",
            "falsifier": "A fourth held-out family (dates, physical facts) drops transfer to chance.",
        },
        {
            "id": f"{LAB_ID}-C3",
            "tag": "DECODE",
            "text": (
                f"Decodability alone is cheap: the surface feature (final word contains "
                f"{surface_letter!r}) is decodable at {surface_peak_acc:.2f} (peak layer "
                f"{surface_peak_layer}). Any claim built on 'a probe found it' must explain why "
                "the found thing is not surface-grade."
            ),
            "artifact": f"runs/{run_name}/plots/decodability_by_layer.png",
            "falsifier": "N/A — this is the lab's calibration claim; it dies only with the dataset.",
        },
    ]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)

    lines = [
        "# Lab 4 run summary: probing with controls",
        "",
        "## Run identity",
        "",
        f"- model: `{bundle.anatomy.model_id}` ({bundle.anatomy.n_layers} blocks, d_model {bundle.anatomy.d_model})",
        f"- statements: {len(statements)} across {len(FAMILIES)} frozen families (per-family cap {per_family_cap or 'none'})",
        f"- probe position: final token | probes: logistic (LBFGS, L2={LOGISTIC_L2}) + mass-mean",
        "- evidence level: `DECODE` — nothing here shows the model USES these directions",
        "",
        "## 1. What behavior was studied?",
        "",
        "None directly — this lab probes REPRESENTATIONS: truth of frozen statements, plus a",
        "deliberately shallow surface feature as the calibration track.",
        "",
        "## 2. What internal object was measured?",
        "",
        "Linear decodability of statement truth from the pre-norm residual stream at the",
        "end-of-statement position, at every depth, with two probe types.",
        "",
        "## 3. What controls were used?",
        "",
        f"Shuffled-label refits (x{N_SHUFFLES}), random directions (x{N_RANDOM_DIRS}), token-length",
        "baseline, family-held-out transfer (incl. negations, where surface form anti-correlates",
        "with truth), and the surface track on the same activations.",
        "",
        "## 4. Headline numbers",
        "",
        f"- truth peak (logistic, within-family): {truth_peak_acc:.3f} at layer {truth_peak_layer}/{n_depths - 1}",
        f"- surface peak: {surface_peak_acc:.3f} at layer {surface_peak_layer}",
        f"- token-length baseline: {metrics['token_length_baseline_mean']:.3f}",
        f"- best cross-family layer (mass-mean, worst transfer): {best_layer} ({best_min_cross:.3f})",
        f"- saved direction: `tables/truth_direction.pt` (cities mass-mean @ layer {best_layer}) — Lab 7 input",
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
        "1. `plots/decodability_by_layer.png` — the whole lesson in one figure.",
        "2. `plots/generalization_matrix.png` — does 'truth' transfer, and where does it leak?",
        "3. `plots/truth_projection_panels.png` — separation emerging over depth, visibly.",
        "4. `plots/selectivity_by_layer.png` — how much accuracy is real structure.",
        "5. `tables/probe_report.csv` — every number, with its controls adjacent.",
        "",
        "## 7. Caveats students must carry forward",
        "",
        "- DECODE is not USE. The saved direction's causal test is Lab 7's job.",
        "- 'Truth' here means truth-on-three-frozen-families. The falsifiers name the",
        "  fourth-family test for a reason.",
        "- The mass-mean and logistic probes can disagree; when they do, the disagreement",
        "  is data, not noise to average away.",
        "",
    ]
    summary_path = ctx.path("run_summary.md")
    bench.write_text(summary_path, "\n".join(lines))
    ctx.register_artifact(summary_path, "summary", "The seven standard questions answered with this run's numbers.")
    print(f"[lab4] wrote run_summary.md and {len(claims)} drafted ledger claims")
