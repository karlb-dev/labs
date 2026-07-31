"""Post-freeze audit of the Phase 3 true-vs-distractor bridge geometry.

The P3-P3 rescue contrast added lexical token rows to the protected span.
This runner replays the frozen Qwen composed cohort and records, for every
layer and sequence position, whether the true and distractor entities added
the same *geometric* protection.  It does not redefine the confirmatory
estimand: the learned geometry adjustment is a post-freeze diagnostic.

Analysis contract (fixed in source before the audit outcomes are read):

* outcome: ``lp_true_bridge - lp_distractor_bridge``;
* geometry features: true-minus-distractor differences in piece count,
  added protected rank, pre-safe selected-span overlap, protected/selected
  ranks, removed energy, lost rank, answer/bridge survival, clean-span
  overlap, activation score, and bridge-answer cosine;
* geometry-only prediction: nested leave-one-family-out ridge regression,
  no intercept, train-fold RMS scaling, lambda grid 1e-4 .. 1e4;
* residual semantic contrast: equal-family mean of outcome minus strictly
  cross-fitted geometry prediction;
* exact-match subset: identical piece count AND identical added-rank vector
  at every audited layer and position.

Each fact is an atomic parquet checkpoint, so a reclaimed VM loses at most
one item.  State headers hash-pin every scientific input and the runner.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from jspace_part2.dictionaries import build_j_dictionaries
from jspace_part2.lib import sha256_file
from jspace_part2.paths import resolve as resolve_uri
from ..ablator3 import Phase3JAblator
from ..bank import load_bank
from ..gpu import require_cuda_gpu
from ..paths3 import metrics_dir
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           resolve_model, write_result3)
from ..scoring import DEFAULT_SPEC, ScoringSession
from ..stats import exact_signflip_test

TIER = "phase3-development"
LAMBDA_GRID = np.logspace(-4, 4, 9)
BOOTSTRAP_SEED = 4242
BOOTSTRAP_DRAWS = 100_000
REPO_DATA = Path(__file__).resolve().parents[2] / "data"


def arg(flag: str, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def piece_ids(tok, entity: str) -> torch.Tensor:
    base = entity.removeprefix("the ").removeprefix("The ").strip()
    ids: set[int] = set()
    for variant in {
            f" {base}", base, f" {base.lower()}", f" {base.title()}"}:
        ids.update(int(i) for i in tok(
            variant, add_special_tokens=False).input_ids)
    if not ids:
        raise ValueError(f"entity {entity!r} tokenizes to no pieces")
    return torch.tensor(sorted(ids), dtype=torch.long)


def validate_state_header(saved: dict, current: dict) -> None:
    mismatches = {
        key: [saved.get(key), value]
        for key, value in current.items()
        if saved.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "refusing incompatible bridge-geometry resume: "
            + json.dumps(mismatches, sort_keys=True))


def _summary(values: pd.Series | np.ndarray) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "median": float(np.median(x)),
        "q025": float(np.quantile(x, 0.025)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "q975": float(np.quantile(x, 0.975)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def _family_inference(frame: pd.DataFrame, value: str) -> dict:
    family = frame.groupby(
        "canonical_family", sort=True)[value].mean()
    values = family.to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for start in range(0, BOOTSTRAP_DRAWS, 10_000):
        n = min(10_000, BOOTSTRAP_DRAWS - start)
        idx = rng.integers(0, len(values), size=(n, len(values)))
        draws[start:start + n] = values[idx].mean(axis=1)
    out = {
        "estimate_equal_family": float(values.mean()),
        "estimate_item_weighted": float(frame[value].mean()),
        "ci95_family_bootstrap": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "n_items": int(len(frame)),
        "n_families": int(len(values)),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    if 3 <= len(values) <= 22:
        out["exact_family_signflip"] = exact_signflip_test(values)
    return out


def _fit_ridge_no_intercept(
        x: np.ndarray, y: np.ndarray, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(np.mean(x * x, axis=0))
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    z = x / scale
    gram = z.T @ z + float(lam) * np.eye(z.shape[1])
    beta = np.linalg.solve(gram, z.T @ y)
    return beta, scale


def nested_family_ridge(
        x: np.ndarray, y: np.ndarray, families: np.ndarray
) -> tuple[np.ndarray, list[dict]]:
    """Strictly cross-fitted geometry-only predictions.

    The held-out family never contributes to scaling, lambda selection, or
    coefficient fitting.  Inner validation also holds out whole families.
    """
    pred = np.full(len(y), np.nan)
    choices: list[dict] = []
    for outer in sorted(set(families.tolist())):
        test = families == outer
        train = ~test
        inner_families = sorted(set(families[train].tolist()))
        losses = {}
        for lam in LAMBDA_GRID:
            sqerr = []
            for inner in inner_families:
                val = train & (families == inner)
                fit = train & (families != inner)
                beta, scale = _fit_ridge_no_intercept(
                    x[fit], y[fit], float(lam))
                inner_pred = (x[val] / scale) @ beta
                sqerr.extend(((y[val] - inner_pred) ** 2).tolist())
            losses[float(lam)] = float(np.mean(sqerr))
        chosen = min(losses, key=lambda value: (losses[value], value))
        beta, scale = _fit_ridge_no_intercept(x[train], y[train], chosen)
        pred[test] = (x[test] / scale) @ beta
        choices.append({
            "held_out_family": outer,
            "lambda": float(chosen),
            "inner_mse_by_lambda": {
                f"{value:.4g}": losses[float(value)]
                for value in LAMBDA_GRID},
        })
    if not np.isfinite(pred).all():
        raise RuntimeError("cross-fitted geometry prediction is incomplete")
    return pred, choices


def analyze(item: pd.DataFrame, site: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    metrics = [
        "protected_rank_before", "protected_rank_after", "added_rank",
        "added_selected_overlap", "rank_selected_before",
        "rank_selected", "removed_energy_l2_sq", "removed_energy_frac",
        "lost_rank", "answer_dir_survival_mean",
        "diagnostic_dir_survival_mean", "diagnostic_base_overlap",
        "diagnostic_activation_score_mean",
        "diagnostic_activation_score_max",
        "diagnostic_answer_cosine_mean",
    ]
    aggregation = {metric: "mean" for metric in metrics}
    arm = site.groupby(
        ["fact_id", "canonical_family", "arm"],
        sort=True, as_index=False).agg(aggregation)
    wide = arm.pivot(
        index=["fact_id", "canonical_family"],
        columns="arm", values=metrics)
    paired = item[[
        "fact_id", "canonical_family", "n_tokens", "lp_baseline",
        "lp_span_safe", "lp_true_bridge", "lp_distractor_bridge",
        "true_piece_count", "distractor_piece_count",
    ]].copy()
    paired["rescue"] = (
        paired["lp_true_bridge"] - paired["lp_distractor_bridge"])
    for metric in metrics:
        paired[f"true_{metric}_mean"] = (
            wide[(metric, "true")].reindex(
                pd.MultiIndex.from_frame(
                    paired[["fact_id", "canonical_family"]])).to_numpy())
        paired[f"distractor_{metric}_mean"] = (
            wide[(metric, "distractor")].reindex(
                pd.MultiIndex.from_frame(
                    paired[["fact_id", "canonical_family"]])).to_numpy())
        paired[f"diff_{metric}_mean"] = (
            paired[f"true_{metric}_mean"]
            - paired[f"distractor_{metric}_mean"])
    paired["diff_piece_count"] = (
        paired.true_piece_count - paired.distractor_piece_count)

    # Strict exact matching on the complete (layer, position) added-rank
    # vector, not merely its mean.
    rank_profiles = {}
    for (fact_id, arm_name), sub in site.groupby(["fact_id", "arm"]):
        ordered = sub.sort_values(["layer", "position"])
        rank_profiles[(fact_id, arm_name)] = tuple(
            int(value) for value in ordered.added_rank)
    paired["exact_piece_count_match"] = (
        paired.true_piece_count == paired.distractor_piece_count)
    paired["exact_added_rank_profile_match"] = [
        rank_profiles[(fact_id, "true")]
        == rank_profiles[(fact_id, "distractor")]
        for fact_id in paired.fact_id]
    paired["exact_geometry_match"] = (
        paired.exact_piece_count_match
        & paired.exact_added_rank_profile_match)

    feature_names = [
        "diff_piece_count",
        "diff_added_rank_mean",
        "diff_added_selected_overlap_mean",
        "diff_protected_rank_after_mean",
        "diff_rank_selected_mean",
        "diff_removed_energy_frac_mean",
        "diff_lost_rank_mean",
        "diff_answer_dir_survival_mean_mean",
        "diff_diagnostic_dir_survival_mean_mean",
        "diff_diagnostic_base_overlap_mean",
        "diff_diagnostic_activation_score_mean_mean",
        "diff_diagnostic_answer_cosine_mean_mean",
    ]
    x = paired[feature_names].to_numpy(dtype=float)
    y = paired.rescue.to_numpy(dtype=float)
    families = paired.canonical_family.to_numpy()
    prediction, choices = nested_family_ridge(x, y, families)
    paired["geometry_prediction_crossfit"] = prediction
    paired["semantic_residual_crossfit"] = y - prediction

    total = float(np.sum((y - y.mean()) ** 2))
    pred_summary = {
        "features": feature_names,
        "contract": {
            "outer_cv": "leave-one-canonical-family-out",
            "inner_cv": "leave-one-canonical-family-out",
            "intercept": False,
            "scaling": "train-fold RMS without centering",
            "lambda_grid": [float(value) for value in LAMBDA_GRID],
        },
        "folds": choices,
        "crossfit_rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
        "crossfit_r2": (
            float(1 - np.sum((y - prediction) ** 2) / total)
            if total > 0 else None),
        "crossfit_correlation": (
            float(np.corrcoef(y, prediction)[0, 1])
            if np.std(prediction) > 0 and np.std(y) > 0 else None),
        "prediction": _summary(prediction),
    }

    paired_differences = {
        "piece_count": _summary(paired.diff_piece_count),
        **{
            metric: _summary(paired[f"diff_{metric}_mean"])
            for metric in metrics
        },
    }
    exact = paired[paired.exact_geometry_match].copy()
    exact_report = {
        "definition": (
            "equal tokenizer-piece count and identical added-rank vector "
            "at every audited layer and position"),
        "n_items": int(len(exact)),
        "n_families": int(exact.canonical_family.nunique()),
        "fraction": float(len(exact) / len(paired)),
        "rescue": (
            _family_inference(exact, "rescue") if len(exact) else None),
        "underpowered": bool(
            len(exact) == 0 or exact.canonical_family.nunique() < 3),
    }
    report = {
        "n_items": int(len(paired)),
        "n_families": int(paired.canonical_family.nunique()),
        "n_site_rows": int(len(site)),
        "paired_true_minus_distractor_geometry": paired_differences,
        "raw_rescue": _family_inference(paired, "rescue"),
        "geometry_only_prediction": pred_summary,
        "residualized_semantic_contrast": _family_inference(
            paired, "semantic_residual_crossfit"),
        "exact_geometry_matched_subset": exact_report,
        "interpretation_guardrail": (
            "Post-freeze diagnostic only. Cross-fitted adjustment can "
            "quantify predictability by measured geometry but cannot turn "
            "the chosen distractor into a randomized semantic control."),
    }
    return report, paired


def make_figure(paired: pd.DataFrame, path_png: Path, path_pdf: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.5))
    specs = [
        ("diff_piece_count", "True − distractor piece count"),
        ("diff_added_rank_mean", "True − distractor mean added rank"),
        ("diff_added_selected_overlap_mean",
         "True − distractor selected-span overlap"),
    ]
    for axis, (column, label) in zip(axes.flat[:3], specs):
        axis.hist(paired[column], bins=15, color="#355c7d", alpha=0.85)
        axis.axvline(0, color="black", lw=1, ls="--")
        axis.set_xlabel(label)
        axis.set_ylabel("facts")
    axis = axes.flat[3]
    axis.scatter(
        paired.geometry_prediction_crossfit, paired.rescue,
        c=paired.exact_geometry_match.map({True: "#2a9d8f", False: "#c65d47"}),
        alpha=0.8, s=28)
    lo = float(min(axis.get_xlim()[0], axis.get_ylim()[0]))
    hi = float(max(axis.get_xlim()[1], axis.get_ylim()[1]))
    axis.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--")
    axis.set_xlim(lo, hi)
    axis.set_ylim(lo, hi)
    axis.set_xlabel("Cross-fitted geometry-only prediction (nats)")
    axis.set_ylabel("Observed rescue (nats)")
    fig.suptitle("Phase 3 Qwen bridge-protection geometry audit")
    fig.tight_layout()
    for path in (path_png, path_pdf):
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        fig.savefig(tmp, dpi=180, bbox_inches="tight", format=path.suffix[1:])
        os.replace(tmp, path)
    plt.close(fig)


@torch.no_grad()
def main() -> None:  # noqa: C901
    cfg_path = Path(arg("--config"))
    cfg = yaml.safe_load(cfg_path.read_text())
    require_clean_tree("--allow-dirty" in sys.argv)
    slug = cfg["slug"]
    evidence_id = cfg["evidence_id"]

    partition_path = Path(resolve_uri(cfg["partition_uri"]))
    primary_path = Path(resolve_uri(cfg["primary_parquet_uri"]))
    bank_paths = [REPO_DATA / name for name in cfg["banks"]]
    lens_path = Path(resolve_uri(cfg["lens_uri"]))
    model_path = Path(resolve_uri(cfg["model_uri"], must_exist=True))
    header = {
        "schema": "p3-bridge-geometry-state-v1",
        "runner_sha256": sha256_file(__file__),
        "config_sha256": sha256_file(cfg_path),
        "partition_sha256": sha256_file(partition_path),
        "primary_parquet_sha256": sha256_file(primary_path),
        "bank_sha256": {
            path.name: sha256_file(path) for path in bank_paths},
        "lens_sha256": sha256_file(lens_path),
        "model": resolve_model(str(model_path)),
        "slug": slug,
    }

    out_dir = metrics_dir(slug) / "release_audit" / "bridge_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(exist_ok=True)
    state_path = out_dir / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        validate_state_header(state["header"], header)
    else:
        state = {"header": header, "done": {}, "started_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        atomic_json(state_path, state)

    gpu = require_cuda_gpu()
    log("GPU gate PASS: " + json.dumps(gpu, sort_keys=True))
    state["gpu"] = gpu
    atomic_json(state_path, state)

    import jlens
    import transformers
    from jlens import JacobianLens

    tok = transformers.AutoTokenizer.from_pretrained(str(model_path))
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=torch.bfloat16).to("cuda").eval()
    if not next(hf.parameters()).is_cuda:
        raise RuntimeError("model parameters are not on CUDA")
    model = jlens.from_hf(hf, tok)
    sess = ScoringSession(tok, DEFAULT_SPEC, device="cuda")
    lens = JacobianLens.load(str(lens_path))
    band = [int(value) for value in cfg["band"]]
    k = int(cfg["k"])
    protect_k = int(cfg["protect_top_k"])
    dictionaries = build_j_dictionaries(hf, lens, band)
    ablator = Phase3JAblator(model.layers, band)

    partition = json.loads(partition_path.read_text())["payload"]
    confirmatory = set(partition["confirmatory"])
    primary = pd.read_parquet(primary_path)
    primary = primary[
        (primary.variant == "composed")
        & primary.lp_true_bridge.notna()
        & primary.lp_distractor_bridge.notna()].copy()
    if len(primary) != int(cfg["expected_items"]):
        raise RuntimeError(
            f"expected {cfg['expected_items']} frozen bridge rows, "
            f"found {len(primary)}")
    if set(primary.canonical_family) - confirmatory:
        raise RuntimeError("primary bridge rows escape confirmatory partition")
    bundle_by_id = {
        bundle.fact_id: bundle
        for path in bank_paths for bundle in load_bank(path)}
    if set(primary.fact_id) - set(bundle_by_id):
        raise RuntimeError("frozen primary facts absent from configured banks")

    def run_arm(full, protect_sets, *, base=None, diagnostic=None,
                answer=None, record=False):
        ablator.log = type(ablator.log)()
        ablator.phase, ablator.forward_index = "prefill", 0
        ablator.mode = {
            "dicts": dictionaries, "k": k, "nonneg": True,
            "protect_sets": protect_sets,
            "active_phases": {"prefill"},
            "span_safe": True,
            "record_overlap": record,
            "answer_id": None,
            "answer_ids": answer,
            "base_protect_sets": base,
            "diagnostic_ids": diagnostic,
        }
        with ablator:
            logits = hf(input_ids=full, use_cache=False).logits[0].float()
        ablator.mode = None
        return logits, ablator.log

    tolerance = float(cfg.get("replay_tolerance", 0.002))
    started = time.time()
    primary = primary.sort_values("item_id")
    for ordinal, frozen in enumerate(primary.itertuples(), start=1):
        if frozen.fact_id in state["done"]:
            continue
        bundle = bundle_by_id[frozen.fact_id]
        if not bundle.counterfactual_bridge:
            raise RuntimeError(
                f"{bundle.fact_id} lacks counterfactual bridge")
        alias = bundle.accepted_answers[0]
        full, n_prompt = sess.full_ids(bundle.prompts["composed"], alias)
        ablator.mode = None
        clean = hf(input_ids=full, use_cache=False).logits[0].float()
        base_protect = clean.topk(protect_k, dim=-1).indices
        lp_base = sess.answer_seq_lp(full, clean.cpu(), n_prompt)
        if abs(lp_base - float(frozen.lp_baseline)) > tolerance:
            raise RuntimeError(
                f"baseline replay gate failed for {bundle.fact_id}: "
                f"{lp_base} vs {frozen.lp_baseline}")
        span_logits, _ = run_arm(full, base_protect)
        lp_span = sess.answer_seq_lp(
            full, span_logits.cpu(), n_prompt)
        if abs(lp_span - float(frozen.lp_meanJ_span_safe)) > tolerance:
            raise RuntimeError(
                f"span-safe replay gate failed for {bundle.fact_id}: "
                f"{lp_span} vs {frozen.lp_meanJ_span_safe}")

        answer = sess.answer_ids(alias)[0]
        definitions = {
            "true": piece_ids(tok, bundle.bridge),
            "distractor": piece_ids(tok, bundle.counterfactual_bridge),
        }
        order_rng = np.random.default_rng(
            int(hashlib.sha256(
                f"bridge-geometry-order|{bundle.fact_id}".encode()
            ).hexdigest()[:16], 16))
        arm_order = [
            list(definitions)[index]
            for index in order_rng.permutation(len(definitions))]
        part_rows = []
        lp_by_arm = {}
        for arm_name in arm_order:
            diagnostic = definitions[arm_name].to(base_protect.device)
            after = torch.cat([
                base_protect,
                diagnostic.unsqueeze(0).expand(full.shape[1], -1),
            ], dim=1)
            logits, audit_log = run_arm(
                full, after, base=base_protect,
                diagnostic=diagnostic, answer=answer, record=True)
            lp = sess.answer_seq_lp(full, logits.cpu(), n_prompt)
            lp_by_arm[arm_name] = lp
            frozen_lp = float(
                frozen.lp_true_bridge if arm_name == "true"
                else frozen.lp_distractor_bridge)
            if abs(lp - frozen_lp) > tolerance:
                raise RuntimeError(
                    f"{arm_name} replay gate failed for {bundle.fact_id}: "
                    f"{lp} vs {frozen_lp}")
            positions = {
                (row.layer, row.phase, row.forward_index, row.position): row
                for row in audit_log.positions}
            if len(positions) != len(audit_log.overlap):
                raise RuntimeError("position/overlap audit rows misalign")
            for overlap in audit_log.overlap:
                key = (overlap.layer, overlap.phase,
                       overlap.forward_index, overlap.position)
                position = positions[key]
                row = dataclasses.asdict(overlap)
                row.update({
                    "fact_id": bundle.fact_id,
                    "canonical_family": bundle.canonical_family,
                    "relation_group": bundle.relation_group,
                    "bank": bundle.bank,
                    "arm": arm_name,
                    "n_piece_ids": int(len(diagnostic)),
                    "n_tokens": int(full.shape[1]),
                    "requested_k": int(position.requested_k),
                    "available_positive": int(position.available_positive),
                    "selected_k": int(position.selected_k),
                    "protected_blocked": int(
                        position.protected_blocked),
                })
                part_rows.append(row)

        part_name = (
            f"{ordinal:04d}_"
            f"{hashlib.sha256(bundle.fact_id.encode()).hexdigest()[:16]}"
            ".parquet")
        part_path = parts_dir / part_name
        atomic_parquet(part_path, pd.DataFrame(part_rows))
        state["done"][bundle.fact_id] = {
            "part": part_name,
            "item_id": frozen.item_id,
            "canonical_family": bundle.canonical_family,
            "n_tokens": int(full.shape[1]),
            "lp_baseline": lp_base,
            "lp_span_safe": lp_span,
            "lp_true_bridge": lp_by_arm["true"],
            "lp_distractor_bridge": lp_by_arm["distractor"],
            "true_piece_count": int(len(definitions["true"])),
            "distractor_piece_count": int(len(definitions["distractor"])),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        atomic_json(state_path, state)
        done = len(state["done"])
        rate = (time.time() - started) / max(
            done - (int(cfg["expected_items"]) - len(primary)), 1)
        log(f"{done}/{len(primary)} {bundle.fact_id} "
            f"(this process {time.time() - started:.0f}s)")

    if len(state["done"]) != len(primary):
        raise RuntimeError("audit ended without every frozen item")
    part_paths = [parts_dir / value["part"]
                  for _, value in sorted(state["done"].items())]
    if not all(path.exists() for path in part_paths):
        raise RuntimeError("state names a missing atomic part")
    site = pd.concat(
        [pd.read_parquet(path) for path in part_paths], ignore_index=True)
    item = pd.DataFrame(
        [{"fact_id": fact_id, **value}
         for fact_id, value in sorted(state["done"].items())])
    site_path = out_dir / f"p3_bridge_geometry_sites_{slug}.parquet"
    item_path = out_dir / f"p3_bridge_geometry_items_{slug}.parquet"
    atomic_parquet(site_path, site)
    atomic_parquet(item_path, item)

    analysis, paired = analyze(item, site)
    paired_path = out_dir / f"p3_bridge_geometry_paired_{slug}.parquet"
    atomic_parquet(paired_path, paired)
    figure_png = out_dir / f"p3_bridge_geometry_{slug}.png"
    figure_pdf = out_dir / f"p3_bridge_geometry_{slug}.pdf"
    make_figure(paired, figure_png, figure_pdf)

    analysis["gpu"] = gpu
    analysis["replay_gates"] = {
        "tolerance_nats": tolerance,
        "baseline": "PASS",
        "span_safe": "PASS",
        "true_bridge": "PASS",
        "distractor_bridge": "PASS",
    }
    analysis["state_header"] = header
    command = (
        "python -m jspace_phase3.experiments.p3_bridge_geometry_audit "
        f"--config {cfg_path}")
    result_path = out_dir / f"p3_bridge_geometry_{slug}.json"
    write_result3(
        analysis, result_path,
        Provenance3(
            evidence_id=evidence_id, tier=TIER, command=command,
            config_path=str(cfg_path),
            inputs=header, model=resolve_model(str(model_path)),
            seed=BOOTSTRAP_SEED))
    markdown_path = out_dir / f"p3_bridge_geometry_{slug}.md"
    raw = analysis["raw_rescue"]
    residual = analysis["residualized_semantic_contrast"]
    exact = analysis["exact_geometry_matched_subset"]
    markdown_path.write_text(
        "# Phase 3 Qwen bridge-geometry audit\n\n"
        f"- Frozen composed facts: {analysis['n_items']} across "
        f"{analysis['n_families']} families.\n"
        f"- Raw true-minus-distractor rescue: "
        f"{raw['estimate_equal_family']:+.4f} nats, 95% family bootstrap "
        f"[{raw['ci95_family_bootstrap'][0]:+.4f}, "
        f"{raw['ci95_family_bootstrap'][1]:+.4f}].\n"
        f"- Cross-fitted geometry-adjusted residual: "
        f"{residual['estimate_equal_family']:+.4f} nats, 95% family "
        f"bootstrap [{residual['ci95_family_bootstrap'][0]:+.4f}, "
        f"{residual['ci95_family_bootstrap'][1]:+.4f}].\n"
        f"- Strict piece-count + per-site added-rank matches: "
        f"{exact['n_items']}/{analysis['n_items']} "
        f"({exact['n_families']} families).\n"
        f"- Geometry-only cross-fit R²: "
        f"{analysis['geometry_only_prediction']['crossfit_r2']}.\n\n"
        "This is a post-freeze diagnostic. It does not create an "
        "untouched-family replication and cannot upgrade P3-P3 beyond "
        "the confirmatory true-versus-chosen-distractor result.\n")

    outputs = [
        result_path, markdown_path, site_path, item_path, paired_path,
        figure_png, figure_pdf, state_path,
    ]
    if "--no-register" not in sys.argv:
        register(
            evidence_id, tier=TIER, command=command,
            what=(
                "Post-freeze per-layer/per-position geometry audit of "
                f"the frozen Qwen P3-P3 bridge rescue ({len(item)} facts); "
                "nested family-cross-fitted geometry prediction and strict "
                "piece-count/added-rank matched subset."),
            outputs=outputs, inputs=header)
    log("sealed bridge-geometry audit")
    print(json.dumps({
        "raw_rescue": raw,
        "residualized_semantic_contrast": residual,
        "exact_geometry_matched_subset": exact,
        "geometry_only_prediction": {
            key: analysis["geometry_only_prediction"][key]
            for key in ("crossfit_rmse", "crossfit_r2",
                        "crossfit_correlation")},
    }, indent=1))


if __name__ == "__main__":
    main()
