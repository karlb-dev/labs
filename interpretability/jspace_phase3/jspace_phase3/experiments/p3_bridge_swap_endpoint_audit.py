"""Counterfactual-answer endpoint for the Phase 3 Qwen bridge swap.

The original bridge-mediation run measured loss of the *original* answer
under counterfactual injection.  Disruption alone does not show that the
model moved toward the intended counterfactual.  This post-freeze development
audit scores both answer sets and greedy generations under one fixed,
prompt-only intervention state.

Protocol choices fixed before outcomes:

* intervene on the prompt prefill only, then turn hooks off;
* score original and counterfactual continuations from clones of that same
  ablated prompt KV state;
* primary endpoint is ``lp(cf canonical) - lp(original canonical)``;
* alias-max preference is a sensitivity (never logsumexp, because the frozen
  alias sets contain prefix-overlapping alternatives);
* compare counterfactual bridge injection with baseline, true re-injection,
  geometry-selected unrelated injection, orthogonal random injection, and
  direct counterfactual-answer injection;
* greedy grading uses exact normalized answer boundaries, not unsafe string
  prefix matching.

One atomic parquet is written per fact.  The process hard-gates CUDA before
model load and refuses incompatible resume state.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import re
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
from ..bank import FactBundle, load_bank
from ..gpu import require_cuda_gpu
from ..paths3 import metrics_dir
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           resolve_model, write_result3)
from ..scoring import DEFAULT_SPEC, ScoringSession
from ..seeds import stable_seed
from ..stats import family_signflip_test

TIER = "phase3-development"
SEED = 4242
BOOTSTRAP_DRAWS = 100_000
REPO_DATA = Path(__file__).resolve().parents[2] / "data"


def arg(flag: str, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def validate_state_header(saved: dict, current: dict) -> None:
    mismatch = {
        key: [saved.get(key), value]
        for key, value in current.items()
        if saved.get(key) != value
    }
    if mismatch:
        raise RuntimeError(
            "refusing incompatible semantic-swap resume: "
            + json.dumps(mismatch, sort_keys=True))


def piece_ids(tok, entity: str) -> torch.Tensor:
    base = entity.removeprefix("the ").removeprefix("The ").strip()
    ids: set[int] = set()
    for variant in {
            f" {base}", base, f" {base.lower()}", f" {base.title()}"}:
        ids.update(int(value) for value in tok(
            variant, add_special_tokens=False).input_ids)
    if not ids:
        raise ValueError(f"entity {entity!r} tokenizes to no pieces")
    return torch.tensor(sorted(ids), dtype=torch.long)


def boundary_generation_category(
        generated: str, original_aliases: list[str],
        counterfactual_aliases: list[str]) -> dict:
    """Boundary-safe, deterministic original/counterfactual grading."""
    normalized = DEFAULT_SPEC.normalize(generated)

    def hits(aliases: list[str]) -> list[str]:
        found = []
        for alias in aliases:
            target = DEFAULT_SPEC.normalize(alias)
            if target and (
                    normalized == target
                    or normalized.startswith(target + " ")):
                found.append(alias)
        return found

    original = hits(original_aliases)
    counterfactual = hits(counterfactual_aliases)
    if original and not counterfactual:
        category = "original"
    elif counterfactual and not original:
        category = "counterfactual"
    elif original and counterfactual:
        category = "ambiguous"
    else:
        category = "other"
    return {
        "category": category,
        "normalized": normalized,
        "original_hits": original,
        "counterfactual_hits": counterfactual,
    }


def clone_past_key_values(past):
    """Clone a HF cache so candidate scoring cannot mutate sibling arms."""
    try:
        return copy.deepcopy(past)
    except Exception as error:  # pragma: no cover - version-specific guard
        if isinstance(past, (tuple, list)):
            return type(past)(
                clone_past_key_values(value) for value in past)
        if torch.is_tensor(past):
            return past.clone()
        raise RuntimeError(
            f"cannot clone {type(past).__name__} KV cache") from error


@torch.no_grad()
def continuation_lp(hf, last_logits: torch.Tensor, past,
                    answer_ids: torch.Tensor, *,
                    prompt_length: int) -> float:
    """Score one continuation from an immutable prefill state."""
    tokens = answer_ids.to(last_logits.device).long()
    if tokens.dim() != 1 or not tokens.numel():
        raise ValueError("answer_ids must be a nonempty vector")
    cache = clone_past_key_values(past)
    total = torch.log_softmax(
        last_logits.float(), dim=-1)[tokens[0]]
    for index in range(1, len(tokens)):
        # Hybrid attention/cache models (including Qwen3.6) require the
        # full growing mask here.  A length-1 implicit mask can silently
        # change cached continuation scores relative to full-sequence
        # teacher forcing.
        attention_mask = torch.ones(
            (1, prompt_length + index),
            dtype=torch.long, device=last_logits.device)
        out = hf(
            input_ids=tokens[index - 1].reshape(1, 1),
            attention_mask=attention_mask,
            past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        total = total + torch.log_softmax(
            out.logits[0, -1].float(), dim=-1)[tokens[index]]
    return float(total.cpu())


@torch.no_grad()
def greedy_from_prefill(hf, tok, last_logits: torch.Tensor, past,
                        *, prompt_length: int,
                        max_new_tokens: int) -> tuple[str, list[int]]:
    cache = clone_past_key_values(past)
    token = int(last_logits.argmax())
    generated = [token]
    eos = {
        int(value) for value in (
            tok.eos_token_id
            if isinstance(tok.eos_token_id, list)
            else [tok.eos_token_id])
        if value is not None}
    for _ in range(max_new_tokens - 1):
        if token in eos:
            break
        attention_mask = torch.ones(
            (1, prompt_length + len(generated)),
            dtype=torch.long, device=last_logits.device)
        out = hf(
            input_ids=torch.tensor(
                [[token]], device=last_logits.device),
            attention_mask=attention_mask,
            past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        token = int(out.logits[0, -1].argmax())
        generated.append(token)
    return tok.decode(generated, skip_special_tokens=True), generated


def _unit_mean(dictionary: torch.Tensor,
               ids: torch.Tensor) -> tuple[torch.Tensor, float]:
    mean = dictionary[ids.to(dictionary.device)].float().mean(dim=0)
    norm = float(mean.norm().cpu())
    return torch.nn.functional.normalize(mean, dim=0), norm


def select_unrelated_match(
        tok, dictionaries: dict[int, torch.Tensor], band: list[int],
        bundle: FactBundle, cohort: list[FactBundle],
        original_answer_ids: torch.Tensor,
        counterfactual_answer_ids: torch.Tensor) -> tuple[FactBundle, dict]:
    """Select without outcomes, lexicographically by frozen geometry.

    Exact tokenizer-piece count is prioritized.  Remaining ties minimize
    layerwise signed answer-overlap and raw mean-direction-norm mismatch to
    the counterfactual bridge.  Candidates from the same fact/family are
    excluded and the complete match report is returned.
    """
    target_ids = piece_ids(tok, bundle.counterfactual_bridge)
    target_profiles = []
    for layer in band:
        target, target_norm = _unit_mean(
            dictionaries[layer], target_ids)
        original_answer, _ = _unit_mean(
            dictionaries[layer], original_answer_ids)
        counterfactual_answer, _ = _unit_mean(
            dictionaries[layer], counterfactual_answer_ids)
        target_profiles.append({
            "norm": target_norm,
            "orig_cos": float((target @ original_answer).cpu()),
            "cf_cos": float((target @ counterfactual_answer).cpu()),
        })

    candidates = []
    for other in cohort:
        if (other.fact_id == bundle.fact_id
                or other.canonical_family == bundle.canonical_family):
            continue
        candidate_ids = piece_ids(tok, other.bridge)
        overlap_sq = []
        norm_sq = []
        for layer, target in zip(band, target_profiles):
            direction, raw_norm = _unit_mean(
                dictionaries[layer], candidate_ids)
            original_answer, _ = _unit_mean(
                dictionaries[layer], original_answer_ids)
            counterfactual_answer, _ = _unit_mean(
                dictionaries[layer], counterfactual_answer_ids)
            overlap_sq.extend([
                (float((direction @ original_answer).cpu())
                 - target["orig_cos"]) ** 2,
                (float((direction @ counterfactual_answer).cpu())
                 - target["cf_cos"]) ** 2,
            ])
            norm_sq.append((raw_norm - target["norm"]) ** 2)
        candidates.append((
            (
                abs(len(candidate_ids) - len(target_ids)),
                float(np.sqrt(np.mean(overlap_sq))),
                float(np.sqrt(np.mean(norm_sq))),
                other.fact_id,
            ),
            other,
            candidate_ids,
        ))
    if not candidates:
        raise RuntimeError(f"no unrelated candidate for {bundle.fact_id}")
    score, selected, selected_ids = min(candidates, key=lambda value: value[0])
    report = {
        "target_entity": bundle.counterfactual_bridge,
        "target_piece_count": int(len(target_ids)),
        "selected_fact_id": selected.fact_id,
        "selected_family": selected.canonical_family,
        "selected_entity": selected.bridge,
        "selected_piece_count": int(len(selected_ids)),
        "piece_count_difference": int(score[0]),
        "piece_count_exact": bool(score[0] == 0),
        "answer_overlap_rmse": float(score[1]),
        "raw_mean_direction_norm_rmse": float(score[2]),
        "injected_direction_norm": 1.0,
        "removed_energy_match": (
            "exact by shared bridge-only lesion and per-position "
            "energy-matched injection scale"),
        "selection_used_outcomes": False,
    }
    return selected, report


def orthogonal_random_directions(
        dictionaries: dict[int, torch.Tensor], band: list[int],
        true_ids: torch.Tensor, counterfactual_ids: torch.Tensor,
        fact_id: str) -> dict[int, torch.Tensor]:
    output = {}
    ids = torch.unique(torch.cat([true_ids, counterfactual_ids]))
    for layer in band:
        dictionary = dictionaries[layer]
        rows = dictionary[ids.to(dictionary.device)].float()
        u, s, _ = torch.linalg.svd(rows.T, full_matrices=False)
        threshold = (s[:1] * 1e-4).clamp_min(1e-7)
        basis = u * (s > threshold).unsqueeze(0)
        generator = torch.Generator().manual_seed(stable_seed(
            "p3-swap-orthogonal-random",
            f"{fact_id}|layer={layer}", SEED))
        vector = torch.randn(
            rows.shape[1], generator=generator,
            dtype=torch.float32).to(rows.device)
        vector = vector - basis @ (basis.T @ vector)
        if float(vector.norm()) < 1e-6:
            raise RuntimeError("orthogonal random direction collapsed")
        output[layer] = torch.nn.functional.normalize(vector, dim=0)
    return output


def _family_inference(frame: pd.DataFrame, value: str) -> dict:
    family = frame.groupby(
        "canonical_family", sort=True)[value].mean()
    values = family.to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for start in range(0, BOOTSTRAP_DRAWS, 10_000):
        n = min(10_000, BOOTSTRAP_DRAWS - start)
        idx = rng.integers(0, len(values), size=(n, len(values)))
        draws[start:start + n] = values[idx].mean(axis=1)
    result = {
        "estimate_equal_family": float(values.mean()),
        "estimate_item_weighted": float(frame[value].mean()),
        "ci95_family_bootstrap": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "n_items": int(len(frame)),
        "n_families": int(len(values)),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": SEED,
    }
    if len(values) >= 3:
        result["family_signflip"] = family_signflip_test(
            values, draws=100_000, seed=SEED)
    return result


def analyze(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    required_arms = {
        "baseline", "span_safe", "true_protect", "distractor_protect",
        "bridge_only", "cf_swap", "true_reinject", "unrelated_swap",
        "random_orthogonal_swap", "cf_answer_swap",
    }
    if set(frame.arm) != required_arms:
        raise RuntimeError(
            f"semantic-swap arm mismatch: {sorted(set(frame.arm))}")
    id_cols = ["fact_id", "canonical_family"]
    values = [
        "lp_original_canonical", "lp_counterfactual_canonical",
        "preference_canonical", "lp_original_max_alias",
        "lp_counterfactual_max_alias", "preference_max_alias",
    ]
    wide = frame.pivot(index=id_cols, columns="arm", values=values)
    paired = frame[frame.arm == "baseline"][id_cols + [
        "legacy_lp_original", "unrelated_match_json"]].copy()
    pair_index = pd.MultiIndex.from_frame(paired[id_cols])
    expanded = {}
    for value in values:
        baseline = wide[(value, "baseline")].reindex(
            pair_index).to_numpy()
        for arm in sorted(required_arms):
            arm_value = wide[(value, arm)].reindex(
                pair_index).to_numpy()
            expanded[f"{value}__{arm}"] = arm_value
            expanded[f"{value}_shift__{arm}"] = arm_value - baseline
    paired = pd.concat([
        paired.reset_index(drop=True),
        pd.DataFrame(expanded),
    ], axis=1)

    paired["primary_cf_preference_shift"] = (
        paired["preference_canonical_shift__cf_swap"])
    paired["primary_cf_vs_unrelated"] = (
        paired["preference_canonical__cf_swap"]
        - paired["preference_canonical__unrelated_swap"])
    paired["cf_vs_true_reinject"] = (
        paired["preference_canonical__cf_swap"]
        - paired["preference_canonical__true_reinject"])
    paired["cf_vs_random_orthogonal"] = (
        paired["preference_canonical__cf_swap"]
        - paired["preference_canonical__random_orthogonal_swap"])
    paired["cf_vs_cf_answer"] = (
        paired["preference_canonical__cf_swap"]
        - paired["preference_canonical__cf_answer_swap"])

    arm_results = {}
    for arm in sorted(required_arms):
        sub = frame[frame.arm == arm].copy()
        sub["preference_shift"] = (
            sub.preference_canonical
            - paired.set_index("fact_id").loc[
                sub.fact_id, "preference_canonical__baseline"].to_numpy())
        sub["original_lp_shift"] = (
            sub.lp_original_canonical
            - paired.set_index("fact_id").loc[
                sub.fact_id, "lp_original_canonical__baseline"].to_numpy())
        sub["counterfactual_lp_shift"] = (
            sub.lp_counterfactual_canonical
            - paired.set_index("fact_id").loc[
                sub.fact_id,
                "lp_counterfactual_canonical__baseline"].to_numpy())
        generation = {}
        for category in (
                "original", "counterfactual", "other", "ambiguous"):
            sub[f"generation_{category}"] = (
                sub.greedy_category == category).astype(float)
            family_rate = sub.groupby(
                "canonical_family")[f"generation_{category}"].mean()
            generation[category] = {
                "item_rate": float(sub[f"generation_{category}"].mean()),
                "equal_family_rate": float(family_rate.mean()),
                "n": int(sub[f"generation_{category}"].sum()),
            }
        arm_results[arm] = {
            "preference_absolute": _family_inference(
                sub, "preference_canonical"),
            "preference_shift_from_baseline": (
                None if arm == "baseline"
                else _family_inference(sub, "preference_shift")),
            "original_lp_shift_from_baseline": (
                None if arm == "baseline"
                else _family_inference(sub, "original_lp_shift")),
            "counterfactual_lp_shift_from_baseline": (
                None if arm == "baseline"
                else _family_inference(sub, "counterfactual_lp_shift")),
            "greedy_generation": generation,
        }

    match = [
        json.loads(value) for value in
        frame[frame.arm == "baseline"].unrelated_match_json]
    replay_error = np.abs(
        paired["lp_original_canonical__baseline"]
        - paired.legacy_lp_original)
    report = {
        "n_items": int(len(paired)),
        "n_families": int(paired.canonical_family.nunique()),
        "endpoint_contract": {
            "intervention_phase": (
                "prompt prefill only; hooks removed for candidate "
                "continuations and greedy decode"),
            "primary_preference": (
                "lp(counterfactual canonical) - lp(original canonical)"),
            "alias_sensitivity": (
                "max alias LP; logsumexp forbidden because frozen alias "
                "sets contain prefix-overlapping alternatives"),
            "generation_grader": (
                "lower-alphanumeric-space normalization plus exact word "
                "boundary; ambiguous dual hits retained"),
        },
        "arm_results": arm_results,
        "primary_cf_preference_shift": _family_inference(
            paired, "primary_cf_preference_shift"),
        "primary_cf_vs_unrelated_matched_injection": _family_inference(
            paired, "primary_cf_vs_unrelated"),
        "control_contrasts": {
            "cf_swap_minus_true_reinject": _family_inference(
                paired, "cf_vs_true_reinject"),
            "cf_swap_minus_random_orthogonal": _family_inference(
                paired, "cf_vs_random_orthogonal"),
            "cf_swap_minus_cf_answer_direction": _family_inference(
                paired, "cf_vs_cf_answer"),
        },
        "baseline_replay_gate": {
            "max_abs_error_vs_legacy_full_sequence": float(
                replay_error.max()),
            "mean_abs_error": float(replay_error.mean()),
        },
        "unrelated_match_quality": {
            "exact_piece_count_fraction": float(np.mean([
                value["piece_count_exact"] for value in match])),
            "max_piece_count_difference": int(max(
                value["piece_count_difference"] for value in match)),
            "answer_overlap_rmse": {
                "mean": float(np.mean([
                    value["answer_overlap_rmse"] for value in match])),
                "max": float(max(
                    value["answer_overlap_rmse"] for value in match)),
            },
            "all_different_fact_and_family": bool(all(
                value["selected_fact_id"] != fact_id
                and value["selected_family"] != family
                for value, fact_id, family in zip(
                    match, paired.fact_id, paired.canonical_family))),
            "removed_energy_and_injection_norm": (
                "exact by construction for all substitution controls"),
        },
        "interpretation_guardrail": (
            "Post-freeze development endpoint on the existing 40-item "
            "mediation cohort. It tests whether the old disruption moves "
            "answer preference, but is not untouched-family replication."),
    }
    return report, paired


def make_figure(frame: pd.DataFrame, paired: pd.DataFrame,
                png: Path, pdf: Path) -> None:
    import matplotlib.pyplot as plt

    arms = [
        "span_safe", "bridge_only", "true_reinject", "cf_swap",
        "unrelated_swap", "random_orthogonal_swap", "cf_answer_swap",
    ]
    labels = [
        "span-safe", "bridge lesion", "true reinject", "CF bridge",
        "unrelated", "orthogonal", "CF answer",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))
    shifts = [
        paired[f"preference_canonical_shift__{arm}"].to_numpy()
        for arm in arms]
    axes[0].boxplot(shifts, tick_labels=labels, showfliers=False)
    axes[0].axhline(0, color="black", lw=1, ls="--")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Δ [LP(counterfactual) − LP(original)]")
    axes[0].set_title("Canonical preference shift")

    x = paired.preference_canonical__baseline
    y = paired.preference_canonical__cf_swap
    axes[1].scatter(x, y, alpha=0.8, color="#355c7d")
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    axes[1].plot([lo, hi], [lo, hi], color="black", lw=1, ls="--")
    axes[1].set_xlabel("Baseline preference")
    axes[1].set_ylabel("Counterfactual-bridge preference")
    axes[1].set_title("Within-item preference movement")

    categories = ["original", "counterfactual", "other", "ambiguous"]
    colors = ["#2a9d8f", "#e76f51", "#8d99ae", "#e9c46a"]
    bottoms = np.zeros(len(arms))
    for category, color in zip(categories, colors):
        values = np.array([
            float((frame[frame.arm == arm].greedy_category
                   == category).mean())
            for arm in arms])
        axes[2].bar(labels, values, bottom=bottoms,
                    label=category, color=color)
        bottoms += values
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Greedy generation fraction")
    axes[2].set_title("Boundary-safe generation outcomes")
    axes[2].legend(fontsize=8)
    fig.suptitle("Phase 3 Qwen semantic bridge-swap endpoint")
    fig.tight_layout()
    for path in (png, pdf):
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        fig.savefig(tmp, dpi=180, bbox_inches="tight",
                    format=path.suffix[1:])
        os.replace(tmp, path)
    plt.close(fig)


@torch.no_grad()
def main() -> None:  # noqa: C901
    cfg_path = Path(arg("--config"))
    cfg = yaml.safe_load(cfg_path.read_text())
    require_clean_tree("--allow-dirty" in sys.argv)
    slug = cfg["slug"]
    evidence_id = cfg["evidence_id"]
    bank_paths = [REPO_DATA / name for name in cfg["banks"]]
    mediation_path = Path(resolve_uri(cfg["mediation_parquet_uri"]))
    lens_path = Path(resolve_uri(cfg["lens_uri"]))
    model_path = Path(resolve_uri(cfg["model_uri"], must_exist=True))
    header = {
        "schema": "p3-bridge-swap-endpoint-state-v1",
        "runner_sha256": sha256_file(__file__),
        "config_sha256": sha256_file(cfg_path),
        "mediation_parquet_sha256": sha256_file(mediation_path),
        "bank_sha256": {
            path.name: sha256_file(path) for path in bank_paths},
        "lens_sha256": sha256_file(lens_path),
        "model": resolve_model(str(model_path)),
        "slug": slug,
    }
    output_subdir = cfg.get("output_subdir", "bridge_swap_endpoint")
    out_dir = metrics_dir(slug) / "release_audit" / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(exist_ok=True)
    state_path = out_dir / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        validate_state_header(state["header"], header)
    else:
        state = {
            "header": header, "done": {},
            "started_utc": time.strftime(
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
    dictionaries = build_j_dictionaries(hf, lens, band)
    ablator = Phase3JAblator(model.layers, band)
    k = int(cfg["k"])
    protect_k = int(cfg["protect_top_k"])
    max_new = int(cfg["max_new_tokens"])

    all_bundles = [
        bundle for path in bank_paths for bundle in load_bank(path)]
    by_id = {bundle.fact_id: bundle for bundle in all_bundles}
    legacy = pd.read_parquet(mediation_path).sort_values("fact_id")
    if len(legacy) != int(cfg["expected_items"]):
        raise RuntimeError(
            f"expected {cfg['expected_items']} mediation items, "
            f"found {len(legacy)}")
    missing = set(legacy.fact_id) - set(by_id)
    if missing:
        raise RuntimeError(f"mediation facts absent from bank: {missing}")
    cohort = [by_id[fact_id] for fact_id in legacy.fact_id]
    legacy_by_id = legacy.set_index("fact_id")

    def prefill(prompt_ids: torch.Tensor, *,
                protect_sets=None, restrict=None, inject=None):
        ablator.log = type(ablator.log)()
        if protect_sets is None and restrict is None and inject is None:
            ablator.mode = None
            return hf(input_ids=prompt_ids, use_cache=True)
        ablator.phase, ablator.forward_index = "prefill", 0
        ablator.mode = {
            "dicts": dictionaries, "k": k, "nonneg": True,
            "protect_sets": protect_sets,
            "active_phases": {"prefill"},
            "span_safe": True,
            "record_overlap": False,
            "answer_id": None,
            "restrict_sets": restrict,
            "inject_dir": inject,
        }
        with ablator:
            output = hf(input_ids=prompt_ids, use_cache=True)
        ablator.mode = None
        return output

    started = time.time()
    for ordinal, bundle in enumerate(cohort, start=1):
        if bundle.fact_id in state["done"]:
            continue
        if not (bundle.counterfactual_bridge
                and bundle.counterfactual_answer
                and bundle.counterfactual_accepted):
            raise RuntimeError(
                f"incomplete counterfactual for {bundle.fact_id}")
        prompt_ids = sess.prompt_ids(bundle.prompts["composed"])
        original_ids = sess.answer_ids(bundle.accepted_answers[0])[0]
        counterfactual_ids = sess.answer_ids(
            bundle.counterfactual_accepted[0])[0]
        true_ids = piece_ids(tok, bundle.bridge)
        cf_bridge_ids = piece_ids(tok, bundle.counterfactual_bridge)
        unrelated_bundle, unrelated_match = select_unrelated_match(
            tok, dictionaries, band, bundle, cohort,
            original_ids, counterfactual_ids)
        unrelated_ids = piece_ids(tok, unrelated_bundle.bridge)

        clean = prefill(prompt_ids)
        clean_protect = clean.logits[0].topk(
            protect_k, dim=-1).indices

        true_direction = {
            layer: dictionaries[layer][
                true_ids.to(dictionaries[layer].device)].float().mean(0)
            for layer in band}
        cf_direction = {
            layer: dictionaries[layer][
                cf_bridge_ids.to(
                    dictionaries[layer].device)].float().mean(0)
            for layer in band}
        unrelated_direction = {
            layer: dictionaries[layer][
                unrelated_ids.to(
                    dictionaries[layer].device)].float().mean(0)
            for layer in band}
        cf_answer_direction = {
            layer: dictionaries[layer][
                counterfactual_ids.to(
                    dictionaries[layer].device)].float().mean(0)
            for layer in band}
        random_direction = orthogonal_random_directions(
            dictionaries, band, true_ids, cf_bridge_ids, bundle.fact_id)

        def with_extra(extra):
            extra = extra.to(clean_protect.device)
            return torch.cat([
                clean_protect,
                extra.unsqueeze(0).expand(prompt_ids.shape[1], -1),
            ], dim=1)

        arm_specs = {
            "baseline": {},
            "span_safe": {"protect_sets": clean_protect},
            "true_protect": {
                "protect_sets": with_extra(true_ids)},
            "distractor_protect": {
                "protect_sets": with_extra(cf_bridge_ids)},
            "bridge_only": {
                "protect_sets": clean_protect,
                "restrict": true_ids.to(clean_protect.device)},
            "cf_swap": {
                "protect_sets": clean_protect,
                "restrict": true_ids.to(clean_protect.device),
                "inject": cf_direction},
            "true_reinject": {
                "protect_sets": clean_protect,
                "restrict": true_ids.to(clean_protect.device),
                "inject": true_direction},
            "unrelated_swap": {
                "protect_sets": clean_protect,
                "restrict": true_ids.to(clean_protect.device),
                "inject": unrelated_direction},
            "random_orthogonal_swap": {
                "protect_sets": clean_protect,
                "restrict": true_ids.to(clean_protect.device),
                "inject": random_direction},
            "cf_answer_swap": {
                "protect_sets": clean_protect,
                "restrict": true_ids.to(clean_protect.device),
                "inject": cf_answer_direction},
        }
        order_rng = np.random.default_rng(stable_seed(
            "p3-bridge-swap-arm-order", bundle.fact_id, SEED))
        order = [
            list(arm_specs)[index]
            for index in order_rng.permutation(len(arm_specs))]
        rows = []
        for arm in order:
            output = clean if arm == "baseline" else prefill(
                prompt_ids, **arm_specs[arm])
            last_logits = output.logits[0, -1].float()
            original_lps = {
                alias: continuation_lp(
                    hf, last_logits, output.past_key_values,
                    sess.answer_ids(alias)[0],
                    prompt_length=prompt_ids.shape[1])
                for alias in bundle.accepted_answers}
            counterfactual_lps = {
                alias: continuation_lp(
                    hf, last_logits, output.past_key_values,
                    sess.answer_ids(alias)[0],
                    prompt_length=prompt_ids.shape[1])
                for alias in bundle.counterfactual_accepted}
            generated, generated_ids = greedy_from_prefill(
                hf, tok, last_logits, output.past_key_values,
                prompt_length=prompt_ids.shape[1],
                max_new_tokens=max_new)
            grading = boundary_generation_category(
                generated, bundle.accepted_answers,
                bundle.counterfactual_accepted)
            original_canonical = original_lps[bundle.accepted_answers[0]]
            counterfactual_canonical = counterfactual_lps[
                bundle.counterfactual_accepted[0]]
            original_max = max(original_lps.values())
            counterfactual_max = max(counterfactual_lps.values())
            rows.append({
                "fact_id": bundle.fact_id,
                "canonical_family": bundle.canonical_family,
                "relation_group": bundle.relation_group,
                "bank": bundle.bank,
                "arm": arm,
                "lp_original_canonical": original_canonical,
                "lp_counterfactual_canonical": counterfactual_canonical,
                "preference_canonical": (
                    counterfactual_canonical - original_canonical),
                "lp_original_max_alias": original_max,
                "lp_counterfactual_max_alias": counterfactual_max,
                "preference_max_alias": (
                    counterfactual_max - original_max),
                "original_alias_lps_json": json.dumps(
                    original_lps, sort_keys=True),
                "counterfactual_alias_lps_json": json.dumps(
                    counterfactual_lps, sort_keys=True),
                "greedy_text": generated,
                "greedy_token_ids_json": json.dumps(generated_ids),
                "greedy_category": grading["category"],
                "greedy_normalized": grading["normalized"],
                "greedy_original_hits_json": json.dumps(
                    grading["original_hits"]),
                "greedy_counterfactual_hits_json": json.dumps(
                    grading["counterfactual_hits"]),
                "legacy_lp_original": float(
                    legacy_by_id.loc[bundle.fact_id, "lp_base"]),
                "unrelated_match_json": json.dumps(
                    unrelated_match, sort_keys=True),
                "true_piece_count": int(len(true_ids)),
                "counterfactual_bridge_piece_count": int(
                    len(cf_bridge_ids)),
                "counterfactual_answer_piece_count": int(
                    len(counterfactual_ids)),
            })
            if arm != "baseline":
                del output

        baseline_row = next(row for row in rows if row["arm"] == "baseline")
        replay_error = abs(
            baseline_row["lp_original_canonical"]
            - baseline_row["legacy_lp_original"])
        if replay_error > float(cfg["replay_tolerance"]):
            raise RuntimeError(
                f"cache scorer replay gate failed for {bundle.fact_id}: "
                f"{replay_error} nats")
        del clean
        part_name = (
            f"{ordinal:04d}_"
            f"{hashlib.sha256(bundle.fact_id.encode()).hexdigest()[:16]}"
            ".parquet")
        atomic_parquet(parts_dir / part_name, pd.DataFrame(rows))
        state["done"][bundle.fact_id] = {
            "part": part_name,
            "canonical_family": bundle.canonical_family,
            "elapsed_seconds": round(time.time() - started, 3),
            "baseline_replay_abs_error": replay_error,
        }
        atomic_json(state_path, state)
        log(f"{len(state['done'])}/{len(cohort)} {bundle.fact_id} "
            f"({time.time() - started:.0f}s)")

    if len(state["done"]) != len(cohort):
        raise RuntimeError("semantic-swap audit ended incomplete")
    paths = [parts_dir / value["part"]
             for _, value in sorted(state["done"].items())]
    if not all(path.exists() for path in paths):
        raise RuntimeError("semantic-swap state names a missing part")
    frame = pd.concat(
        [pd.read_parquet(path) for path in paths], ignore_index=True)
    frame_path = out_dir / f"p3_bridge_swap_endpoint_{slug}.parquet"
    atomic_parquet(frame_path, frame)
    analysis, paired = analyze(frame)
    if analysis["baseline_replay_gate"][
            "max_abs_error_vs_legacy_full_sequence"] > float(
                cfg["replay_tolerance"]):
        raise RuntimeError("final baseline replay gate failed")
    paired_path = out_dir / f"p3_bridge_swap_endpoint_paired_{slug}.parquet"
    atomic_parquet(paired_path, paired)
    figure_png = out_dir / f"p3_bridge_swap_endpoint_{slug}.png"
    figure_pdf = out_dir / f"p3_bridge_swap_endpoint_{slug}.pdf"
    make_figure(frame, paired, figure_png, figure_pdf)
    analysis["gpu"] = gpu
    analysis["state_header"] = header

    command = (
        "python -m "
        "jspace_phase3.experiments.p3_bridge_swap_endpoint_audit "
        f"--config {cfg_path}")
    result_path = out_dir / f"p3_bridge_swap_endpoint_{slug}.json"
    write_result3(
        analysis, result_path,
        Provenance3(
            evidence_id=evidence_id, tier=TIER, command=command,
            config_path=str(cfg_path), inputs=header,
            model=resolve_model(str(model_path)), seed=SEED))
    markdown_path = out_dir / f"p3_bridge_swap_endpoint_{slug}.md"
    primary = analysis["primary_cf_preference_shift"]
    control = analysis["primary_cf_vs_unrelated_matched_injection"]
    cf_arm = analysis["arm_results"]["cf_swap"]
    markdown_path.write_text(
        "# Phase 3 Qwen counterfactual bridge-swap endpoint\n\n"
        f"- Existing mediation facts: {analysis['n_items']} across "
        f"{analysis['n_families']} families.\n"
        f"- Counterfactual-bridge preference shift from baseline: "
        f"{primary['estimate_equal_family']:+.4f} nats, 95% family "
        f"bootstrap [{primary['ci95_family_bootstrap'][0]:+.4f}, "
        f"{primary['ci95_family_bootstrap'][1]:+.4f}].\n"
        f"- Counterfactual bridge minus geometry-selected unrelated "
        f"injection: {control['estimate_equal_family']:+.4f} nats, "
        f"95% family bootstrap "
        f"[{control['ci95_family_bootstrap'][0]:+.4f}, "
        f"{control['ci95_family_bootstrap'][1]:+.4f}].\n"
        f"- Under counterfactual bridge injection, greedy original / "
        f"counterfactual / other rates: "
        f"{cf_arm['greedy_generation']['original']['item_rate']:.3f} / "
        f"{cf_arm['greedy_generation']['counterfactual']['item_rate']:.3f} "
        f"/ {cf_arm['greedy_generation']['other']['item_rate']:.3f}.\n\n"
        "This post-freeze development audit uses prompt-only intervention "
        "and one shared KV-state contract for both answer candidates. It "
        "does not supply untouched-family replication.\n")
    outputs = [
        result_path, markdown_path, frame_path, paired_path,
        figure_png, figure_pdf, state_path,
    ]
    if "--no-register" not in sys.argv:
        register(
            evidence_id, tier=TIER, command=command,
            what=(
                "Prompt-only semantic endpoint for the Qwen bridge swap: "
                f"{len(paired)} mediation facts, original/counterfactual "
                "canonical and alias-max LP, boundary-safe greedy outcomes, "
                "and five substitution controls."),
            outputs=outputs, inputs=header)
    log("sealed semantic bridge-swap endpoint")
    print(json.dumps({
        "primary_cf_preference_shift": primary,
        "primary_cf_vs_unrelated_matched_injection": control,
        "cf_swap": cf_arm,
        "unrelated_match_quality": analysis["unrelated_match_quality"],
        "baseline_replay_gate": analysis["baseline_replay_gate"],
    }, indent=1))


if __name__ == "__main__":
    main()
