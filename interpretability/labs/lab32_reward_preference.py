"""Lab 32: Reward models and preference circuits.

Tier A uses a DPO-style log-prob-ratio proxy instead of an external reward
model: policy mean log-prob minus a response-unconditional reference baseline.
The lab treats that proxy as a measurement target to audit. A preference signal
only earns support if it beats length, politeness, agreement, sentiment,
hedging, refusal, random-direction, and shuffled-score controls.

Evidence level: ATTR + DECODE + CAUSAL, scoped to benign paired responses and
lightweight activation-addition tests on A/B preference prompts.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import pathlib
import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import interp_bench as bench

LAB_ID = "L32"
DATA_FILE = "preference_circuit_pairs.csv"
PROMPT_SET_CAPS = {"small": 20, "medium": 20, "full": 0}
SCIENCE_READY_MIN_ROWS = 16
STEER_SCALES = (-1.0, 0.0, 1.0)
CONFOUND_FEATURES = ("length", "politeness", "agreement", "sentiment", "hedging", "refusal")
REVIEW_FIELDS = (
    "student_preference",
    "student_confidence",
    "student_rationale_span",
    "reviewer_preference",
    "agreement_status",
)

POLITE_WORDS = {"please", "thanks", "thank", "sorry", "appreciate", "happy", "respectfully"}
AGREEMENT_WORDS = {"yes", "right", "correct", "exactly", "agree", "confirm", "sure", "absolutely"}
POSITIVE_WORDS = {"clear", "good", "great", "approved", "exciting", "useful", "nice", "happy", "correct", "ready"}
NEGATIVE_WORDS = {"wrong", "late", "blocked", "risk", "terrible", "disaster", "stuck", "not", "cannot", "delayed"}
HEDGING_WORDS = {"cannot", "need", "enough", "information", "infer", "depends", "check", "ask", "unknown"}
REFUSAL_WORDS = {"cannot", "do", "not", "without", "consent", "private", "approved", "instead", "directly"}


@dataclasses.dataclass
class PreferencePair:
    pair_id: str
    domain: str
    prompt: str
    response_a: str
    response_b: str
    preferred: str
    preference_type: str
    confound_type: str
    split: str
    notes: str

    @property
    def label(self) -> int:
        return 1 if self.preferred.strip().lower() == "a" else 0

    @property
    def preferred_response(self) -> str:
        return self.response_a if self.label == 1 else self.response_b

    @property
    def rejected_response(self) -> str:
        return self.response_b if self.label == 1 else self.response_a


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rounded(value: Any, digits: int = 4) -> Any:
    try:
        f = float(value)
    except Exception:
        return value
    return round(f, digits) if math.isfinite(f) else ""


def safe_mean(values: Sequence[Any], default: float = float("nan")) -> float:
    vals: list[float] = []
    for value in values:
        try:
            f = float(value)
        except Exception:
            continue
        if math.isfinite(f):
            vals.append(f)
    return float(statistics.fmean(vals)) if vals else default


def safe_corr(xs: Sequence[Any], ys: Sequence[Any]) -> float:
    pairs: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        try:
            xf = float(x)
            yf = float(y)
        except Exception:
            continue
        if math.isfinite(xf) and math.isfinite(yf):
            pairs.append((xf, yf))
    if len(pairs) < 2:
        return float("nan")
    xbar = statistics.fmean(x for x, _ in pairs)
    ybar = statistics.fmean(y for _, y in pairs)
    num = sum((x - xbar) * (y - ybar) for x, y in pairs)
    denx = math.sqrt(sum((x - xbar) ** 2 for x, _ in pairs))
    deny = math.sqrt(sum((y - ybar) ** 2 for _, y in pairs))
    return num / (denx * deny) if denx > 1e-12 and deny > 1e-12 else float("nan")


def auc_binary(labels: Sequence[int], scores: Sequence[float]) -> float:
    pos = [float(s) for y, s in zip(labels, scores) if int(y) == 1 and math.isfinite(float(s))]
    neg = [float(s) for y, s in zip(labels, scores) if int(y) == 0 and math.isfinite(float(s))]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for ps in pos:
        for ns in neg:
            wins += 1.0 if ps > ns else 0.5 if ps == ns else 0.0
    return wins / (len(pos) * len(neg))


def words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*", text.lower())


def signed_preferred_margin(pair: PreferencePair, a_minus_b: float) -> float:
    return float(a_minus_b) if pair.label == 1 else -float(a_minus_b)


def predict_preference(a_minus_b: float) -> str:
    return "a" if float(a_minus_b) >= 0.0 else "b"


def data_path(args: Any) -> pathlib.Path:
    prompt_set = str(getattr(args, "prompt_set", "") or "")
    candidate = pathlib.Path(prompt_set)
    if prompt_set not in PROMPT_SET_CAPS and candidate.suffix.lower() in {".csv", ".tsv"}:
        return candidate if candidate.is_absolute() else (bench.COURSE_ROOT / candidate).resolve()
    return bench.COURSE_ROOT / "data" / DATA_FILE


def load_pairs(ctx: bench.RunContext) -> tuple[list[PreferencePair], dict[str, Any]]:
    path = data_path(ctx.args)
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    pairs = [PreferencePair(**row) for row in rows]
    cap = PROMPT_SET_CAPS.get(str(ctx.args.prompt_set), 0)
    if cap:
        pairs = pairs[:cap]
    if int(ctx.args.max_examples or 0) > 0:
        pairs = pairs[: int(ctx.args.max_examples)]
    info = {
        "data_path": str(path),
        "sha256": file_sha256(path),
        "n_rows_file": len(rows),
        "n_rows_selected": len(pairs),
        "domains": dict(Counter(p.domain for p in pairs)),
        "preference_types": dict(Counter(p.preference_type for p in pairs)),
        "confound_types": dict(Counter(p.confound_type for p in pairs)),
        "splits": dict(Counter(p.split for p in pairs)),
        "science_ready": len(pairs) >= SCIENCE_READY_MIN_ROWS,
        "safety_scope": "benign preference pairs only; no prompt optimization, jailbreak search, or refusal ablation",
        "canonical_tier_a_mode": "dpo_logprob_ratio_proxy_with_unigram_reference",
    }
    return pairs, info


def response_context(pair: PreferencePair) -> str:
    return pair.prompt.rstrip() + "\n\nAssistant:"


def response_text(pair: PreferencePair, which: str) -> str:
    return pair.response_a if which == "a" else pair.response_b


def mean_token_logprob(bundle: bench.ModelBundle, context: str, continuation: str) -> float:
    import torch

    if not continuation:
        return float("nan")
    tok = bundle.tokenizer
    continuation = " " + continuation.strip()
    ctx_ids = tok(context, return_tensors="pt", add_special_tokens=False)["input_ids"]
    full_ids = tok(context + continuation, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if full_ids.shape[1] <= ctx_ids.shape[1]:
        return float("nan")
    ids = full_ids.to(bundle.input_device)
    with torch.no_grad():
        out = bundle.model(input_ids=ids, use_cache=False)
    logprobs = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
    start = max(0, ctx_ids.shape[1] - 1)
    targets = ids[0, ctx_ids.shape[1] :]
    picked = logprobs[start : start + len(targets)].gather(1, targets[:, None].to(logprobs.device))
    return float(picked.mean())


def build_reference_counts(bundle: bench.ModelBundle, pairs: Sequence[PreferencePair]) -> dict[str, Any]:
    counts: Counter[int] = Counter()
    total = 0
    for pair in pairs:
        for which in ("a", "b"):
            ids = bundle.tokenizer.encode(" " + response_text(pair, which).strip(), add_special_tokens=False)
            counts.update(int(i) for i in ids)
            total += len(ids)
    vocab = int(getattr(bundle.tokenizer, "vocab_size", 50257) or 50257)
    return {"counts": counts, "total": total, "vocab": vocab, "alpha": 0.25}


def reference_mean_logprob(bundle: bench.ModelBundle, counts: Mapping[str, Any], response: str) -> float:
    ids = bundle.tokenizer.encode(" " + response.strip(), add_special_tokens=False)
    if not ids:
        return float("nan")
    table: Counter[int] = counts["counts"]
    total = float(counts["total"])
    vocab = float(counts["vocab"])
    alpha = float(counts["alpha"])
    denom = total + alpha * vocab
    vals = [math.log((float(table.get(int(i), 0)) + alpha) / denom) for i in ids]
    return safe_mean(vals)


def response_features(bundle: bench.ModelBundle, response: str) -> dict[str, float]:
    toks = words(response)
    low = response.lower()
    token_len = len(bundle.tokenizer.encode(response, add_special_tokens=False))
    sentiment = sum(t in POSITIVE_WORDS for t in toks) - sum(t in NEGATIVE_WORDS for t in toks)
    return {
        "length": float(token_len),
        "politeness": float(sum(t in POLITE_WORDS for t in toks)),
        "agreement": float(sum(t in AGREEMENT_WORDS for t in toks)),
        "sentiment": float(sentiment),
        "hedging": float(sum(t in HEDGING_WORDS for t in toks) + ("not enough information" in low)),
        "refusal": float(sum(t in REFUSAL_WORDS for t in toks) + ("cannot provide" in low) + ("do not" in low)),
    }


def score_pairs(bundle: bench.ModelBundle, pairs: Sequence[PreferencePair]) -> list[dict[str, Any]]:
    ref_counts = build_reference_counts(bundle, pairs)
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        context = response_context(pair)
        policy_a = mean_token_logprob(bundle, context, pair.response_a)
        policy_b = mean_token_logprob(bundle, context, pair.response_b)
        ref_a = reference_mean_logprob(bundle, ref_counts, pair.response_a)
        ref_b = reference_mean_logprob(bundle, ref_counts, pair.response_b)
        feats_a = response_features(bundle, pair.response_a)
        feats_b = response_features(bundle, pair.response_b)
        policy_margin = policy_a - policy_b
        reference_margin = ref_a - ref_b
        dpo_margin = policy_margin - reference_margin
        row: dict[str, Any] = {
            "pair_id": pair.pair_id,
            "domain": pair.domain,
            "preference_type": pair.preference_type,
            "confound_type": pair.confound_type,
            "split": pair.split,
            "preferred": pair.preferred,
            "dpo_proxy_margin_a_minus_b": rounded(dpo_margin),
            "dpo_proxy_preferred_margin": rounded(signed_preferred_margin(pair, dpo_margin)),
            "dpo_proxy_prediction": predict_preference(dpo_margin),
            "policy_logprob_margin_a_minus_b": rounded(policy_margin),
            "reference_logprob_margin_a_minus_b": rounded(reference_margin),
            "policy_a_mean_logprob": rounded(policy_a),
            "policy_b_mean_logprob": rounded(policy_b),
            "reference_a_mean_logprob": rounded(ref_a),
            "reference_b_mean_logprob": rounded(ref_b),
            "response_a_token_count": feats_a["length"],
            "response_b_token_count": feats_b["length"],
            "notes": pair.notes,
            "student_preference": "",
            "student_confidence": "",
            "student_rationale_span": "",
            "reviewer_preference": "",
            "agreement_status": "",
        }
        for feature in CONFOUND_FEATURES:
            margin = feats_a[feature] - feats_b[feature]
            row[f"{feature}_margin_a_minus_b"] = rounded(margin)
            row[f"{feature}_preferred_margin"] = rounded(signed_preferred_margin(pair, margin))
        rows.append(row)
    return rows


def choose_layers(bundle: bench.ModelBundle) -> tuple[int, int]:
    n_layers = int(bundle.anatomy.n_layers)
    steer_layer = max(0, min(n_layers - 1, n_layers // 2))
    stream_depth = max(1, min(n_layers, steer_layer + 1))
    return steer_layer, stream_depth


def capture_response_vectors(bundle: bench.ModelBundle, pairs: Sequence[PreferencePair], stream_depth: int) -> dict[tuple[str, str], Any]:
    vectors: dict[tuple[str, str], Any] = {}
    for pair in pairs:
        for which in ("a", "b"):
            text = pair.prompt.rstrip() + "\n\nAssistant response:\n" + response_text(pair, which)
            capture = bench.run_with_residual_cache(bundle, text)
            vectors[(pair.pair_id, which)] = capture.streams[stream_depth, -1, :].detach().clone()
    return vectors


def unit_vector(vec: Any) -> Any:
    import torch

    norm = torch.linalg.vector_norm(vec.float())
    if float(norm) <= 1e-9:
        return torch.zeros_like(vec.float())
    return vec.float() / norm


def build_directions(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    pairs: Sequence[PreferencePair],
    pair_rows: Sequence[Mapping[str, Any]],
    vectors: Mapping[tuple[str, str], Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    by_pair = {r["pair_id"]: r for r in pair_rows}
    diffs = [vectors[(p.pair_id, "a")].float() - vectors[(p.pair_id, "b")].float() for p in pairs]
    pref_terms = [(1.0 if p.label == 1 else -1.0) * d for p, d in zip(pairs, diffs)]
    pref_raw = torch.stack(pref_terms).mean(dim=0)
    diff_norm = safe_mean([float(torch.linalg.vector_norm(d)) for d in diffs], default=1.0)
    directions: dict[str, Any] = {"preference_residual_direction": unit_vector(pref_raw)}
    for feature in CONFOUND_FEATURES:
        terms = []
        for pair, diff in zip(pairs, diffs):
            margin = float(by_pair[pair.pair_id][f"{feature}_margin_a_minus_b"] or 0.0)
            if abs(margin) > 1e-9:
                terms.append((1.0 if margin > 0 else -1.0) * diff)
        raw = torch.stack(terms).mean(dim=0) if terms else torch.zeros_like(pref_raw)
        directions[f"{feature}_direction"] = unit_vector(raw)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(getattr(ctx.args, "seed", 0) or 0) + 3200)
    directions["random_direction_control"] = unit_vector(torch.randn(pref_raw.shape, dtype=pref_raw.dtype, generator=gen))
    steer_vectors = {name: vec * float(diff_norm) for name, vec in directions.items()}
    metadata = {
        "stream_depth": choose_layers(bundle)[1],
        "steer_layer": choose_layers(bundle)[0],
        "mean_pair_diff_norm": rounded(diff_norm),
        "direction_norms": {name: rounded(float(torch.linalg.vector_norm(vec))) for name, vec in directions.items()},
        "steer_vector_norms": {name: rounded(float(torch.linalg.vector_norm(vec))) for name, vec in steer_vectors.items()},
    }
    return {"unit": directions, "steer": steer_vectors}, metadata


def direction_margins(
    pairs: Sequence[PreferencePair],
    vectors: Mapping[tuple[str, str], Any],
    directions: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    margins: dict[str, dict[str, float]] = defaultdict(dict)
    for pair in pairs:
        diff = vectors[(pair.pair_id, "a")].float() - vectors[(pair.pair_id, "b")].float()
        for name, direction in directions.items():
            margins[pair.pair_id][name] = float(diff @ direction.float())
    return margins


def attach_direction_scores(pair_rows: list[dict[str, Any]], pairs: Sequence[PreferencePair], margins: Mapping[str, Mapping[str, float]]) -> None:
    by_pair = {p.pair_id: p for p in pairs}
    for row in pair_rows:
        pair = by_pair[str(row["pair_id"])]
        for name, per_pair in margins[str(row["pair_id"])].items():
            row[f"{name}_margin_a_minus_b"] = rounded(per_pair)
            row[f"{name}_preferred_margin"] = rounded(signed_preferred_margin(pair, per_pair))
        pref = float(margins[str(row["pair_id"])]["preference_residual_direction"])
        row["preference_direction_prediction"] = predict_preference(pref)


def summarize_scores(
    pairs: Sequence[PreferencePair],
    pair_rows: Sequence[Mapping[str, Any]],
    score_specs: Sequence[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = [p.label for p in pairs]
    by_pair = {r["pair_id"]: r for r in pair_rows}
    for score_name, column, family in score_specs:
        scores = [float(by_pair[p.pair_id].get(column, 0.0) or 0.0) for p in pairs]
        prefs = [signed_preferred_margin(p, s) for p, s in zip(pairs, scores)]
        accuracy = safe_mean([1.0 if ((s >= 0.0) == (p.label == 1)) else 0.0 for p, s in zip(pairs, scores)], default=0.0)
        rows.append({
            "score_name": score_name,
            "score_family": family,
            "n_pairs": len(pairs),
            "auc": rounded(auc_binary(labels, scores)),
            "accuracy": rounded(accuracy),
            "mean_preferred_margin": rounded(safe_mean(prefs, default=0.0)),
            "median_abs_margin": rounded(statistics.median([abs(s) for s in scores]) if scores else 0.0),
            "length_corr": rounded(safe_corr(scores, [float(by_pair[p.pair_id]["length_margin_a_minus_b"]) for p in pairs])),
            "agreement_corr": rounded(safe_corr(scores, [float(by_pair[p.pair_id]["agreement_margin_a_minus_b"]) for p in pairs])),
            "sentiment_corr": rounded(safe_corr(scores, [float(by_pair[p.pair_id]["sentiment_margin_a_minus_b"]) for p in pairs])),
        })
    return rows


def score_specs() -> list[tuple[str, str, str]]:
    specs = [
        ("dpo_logprob_ratio_proxy", "dpo_proxy_margin_a_minus_b", "proxy"),
        ("policy_logprob_only", "policy_logprob_margin_a_minus_b", "proxy_control"),
        ("preference_residual_direction", "preference_residual_direction_margin_a_minus_b", "residual_direction"),
        ("random_direction_control", "random_direction_control_margin_a_minus_b", "control"),
    ]
    specs += [(f"{f}_confound", f"{f}_margin_a_minus_b", "confound") for f in CONFOUND_FEATURES]
    specs += [(f"{f}_direction", f"{f}_direction_margin_a_minus_b", "confound_direction") for f in CONFOUND_FEATURES]
    return specs


def add_shuffled_control(pair_rows: list[dict[str, Any]]) -> None:
    if not pair_rows:
        return
    dpo_scores = [float(r["dpo_proxy_margin_a_minus_b"] or 0.0) for r in pair_rows]
    for i, row in enumerate(pair_rows):
        row["shuffled_score_control_margin_a_minus_b"] = rounded(dpo_scores[(i + 5) % len(dpo_scores)])
        row["shuffled_score_control_preferred_margin"] = rounded(float(row["shuffled_score_control_margin_a_minus_b"]))


def judge_prompt(pair: PreferencePair) -> str:
    return (
        "Prompt:\n"
        + pair.prompt.strip()
        + "\n\nResponse A:\n"
        + pair.response_a.strip()
        + "\n\nResponse B:\n"
        + pair.response_b.strip()
        + "\n\nWhich response is better? Answer"
    )


def token_id_or_first(bundle: bench.ModelBundle, text: str) -> int:
    ids = bundle.tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise RuntimeError(f"No token ids for {text!r}")
    return int(ids[0])


def run_interventions(
    bundle: bench.ModelBundle,
    pairs: Sequence[PreferencePair],
    steer_layer: int,
    preference_vector: Any,
    random_vector: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    a_id = token_id_or_first(bundle, " A")
    b_id = token_id_or_first(bundle, " B")
    for pair in pairs:
        prompt = judge_prompt(pair)
        for direction_name, vector in (("preference_residual_direction", preference_vector), ("random_direction_control", random_vector)):
            for scale in STEER_SCALES:
                logits = bench.next_token_logits(bundle, prompt, steer=(steer_layer, vector, float(scale)))
                margin = float(logits[a_id] - logits[b_id])
                pref_margin = signed_preferred_margin(pair, margin)
                rows.append({
                    "pair_id": pair.pair_id,
                    "domain": pair.domain,
                    "preference_type": pair.preference_type,
                    "confound_type": pair.confound_type,
                    "direction": direction_name,
                    "scale": scale,
                    "a_logit_minus_b_logit": rounded(margin),
                    "preferred_answer_logit_margin": rounded(pref_margin),
                    "preferred": pair.preferred,
                })
    zero_pref = {
        (r["pair_id"], r["direction"]): float(r["preferred_answer_logit_margin"])
        for r in rows
        if float(r["scale"]) == 0.0
    }
    shifts_by_direction: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if float(r["scale"]) != 1.0:
            continue
        base = zero_pref.get((r["pair_id"], r["direction"]))
        if base is not None:
            shifts_by_direction[str(r["direction"])].append(float(r["preferred_answer_logit_margin"]) - base)
    pref_shift = safe_mean(shifts_by_direction["preference_residual_direction"], default=0.0)
    random_shift = safe_mean(shifts_by_direction["random_direction_control"], default=0.0)
    summary = {
        "mean_preference_direction_shift_at_scale_1": rounded(pref_shift),
        "mean_random_direction_shift_at_scale_1": rounded(random_shift),
        "causal_shift_over_random": rounded(pref_shift - random_shift),
        "n_intervention_pairs": len(pairs),
        "supported": bool(pref_shift - random_shift > 0.05 and pref_shift > 0.0),
    }
    return rows, summary


def build_disagreements(pairs: Sequence[PreferencePair], pair_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_pair = {r["pair_id"]: r for r in pair_rows}
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        row = by_pair[pair.pair_id]
        dpo_pred = str(row["dpo_proxy_prediction"])
        dir_pred = str(row.get("preference_direction_prediction", ""))
        policy_pred = predict_preference(float(row["policy_logprob_margin_a_minus_b"] or 0.0))
        if dpo_pred != pair.preferred or dir_pred != pair.preferred or policy_pred != dpo_pred or pair.preference_type == "anti_sycophancy":
            rows.append({
                "pair_id": pair.pair_id,
                "domain": pair.domain,
                "preference_type": pair.preference_type,
                "confound_type": pair.confound_type,
                "preferred": pair.preferred,
                "dpo_proxy_prediction": dpo_pred,
                "policy_logprob_prediction": policy_pred,
                "preference_direction_prediction": dir_pred,
                "dpo_preferred_margin": row["dpo_proxy_preferred_margin"],
                "preference_direction_preferred_margin": row.get("preference_residual_direction_preferred_margin", ""),
                "agreement_preferred_margin": row["agreement_preferred_margin"],
                "reason": "sycophancy_risk" if pair.preference_type == "anti_sycophancy" else "proxy_or_direction_disagreement",
            })
    return rows


def build_evidence(
    report: Sequence[Mapping[str, Any]],
    intervention_summary: Mapping[str, Any],
    data_info: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_name = {r["score_name"]: r for r in report}
    confound_rows = [r for r in report if r["score_family"] in {"confound", "confound_direction"}]
    control_rows = [r for r in report if r["score_family"] == "control" or str(r["score_name"]).endswith("_control")]
    max_confound_auc = max([float(r["auc"] or 0.0) for r in confound_rows], default=0.0)
    max_control_auc = max([float(r["auc"] or 0.0) for r in control_rows], default=0.0)

    def posture(name: str, min_auc: float = 0.65) -> str:
        row = by_name[name]
        auc = float(row["auc"] or 0.0)
        acc = float(row["accuracy"] or 0.0)
        if auc >= min_auc and acc >= 0.60 and auc >= max_confound_auc + 0.05 and auc >= max_control_auc + 0.05:
            return "beats_confound_controls"
        if auc >= min_auc and acc >= 0.60:
            return "predictive_but_confound_limited"
        return "not_supported_on_this_suite"

    rows = []
    for name, rung in (
        ("dpo_logprob_ratio_proxy", "ATTR"),
        ("preference_residual_direction", "DECODE"),
        ("policy_logprob_only", "ATTR_CONTROL"),
        ("random_direction_control", "CONTROL"),
    ):
        row = by_name[name]
        rows.append({
            "method": name,
            "evidence_rung": rung,
            "n_pairs": row["n_pairs"],
            "auc": row["auc"],
            "accuracy": row["accuracy"],
            "mean_preferred_margin": row["mean_preferred_margin"],
            "max_confound_auc": rounded(max_confound_auc),
            "max_control_auc": rounded(max_control_auc),
            "margin_over_best_confound_auc": rounded(float(row["auc"] or 0.0) - max_confound_auc),
            "claim_posture": "control_sanity_check" if "control" in name else posture(name),
        })
    causal_supported = bool(intervention_summary.get("supported"))
    rows.append({
        "method": "activation_addition_preference_direction",
        "evidence_rung": "CAUSAL",
        "n_pairs": intervention_summary["n_intervention_pairs"],
        "auc": "",
        "accuracy": "",
        "mean_preferred_margin": intervention_summary["mean_preference_direction_shift_at_scale_1"],
        "max_confound_auc": rounded(max_confound_auc),
        "max_control_auc": rounded(max_control_auc),
        "margin_over_best_confound_auc": "",
        "claim_posture": "causal_shift_supported" if causal_supported else "causal_shift_not_established",
    })
    supported = sum(1 for r in rows if str(r["claim_posture"]) in {"beats_confound_controls", "causal_shift_supported"})
    metrics = {
        "science_ready": bool(data_info["science_ready"]),
        "n_pairs": data_info["n_rows_selected"],
        "max_confound_auc": rounded(max_confound_auc),
        "max_control_auc": rounded(max_control_auc),
        "dpo_proxy_auc": by_name["dpo_logprob_ratio_proxy"]["auc"],
        "preference_direction_auc": by_name["preference_residual_direction"]["auc"],
        "causal_shift_over_random": intervention_summary["causal_shift_over_random"],
        "supported_evidence_rows": supported,
        "best_non_control_score": max(
            [r for r in report if r["score_family"] not in {"control", "proxy_control"}],
            key=lambda r: float(r["auc"] or 0.0),
        )["score_name"],
    }
    return rows, metrics


def write_tables(
    ctx: bench.RunContext,
    pair_rows: Sequence[Mapping[str, Any]],
    report: Sequence[Mapping[str, Any]],
    disagreements: Sequence[Mapping[str, Any]],
    interventions: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
) -> None:
    specs = [
        ("tables/preference_pair_scores.csv", pair_rows, "Pair-level DPO proxy, policy, reference, confound, and direction margins."),
        ("tables/preference_probe_report.csv", report, "Scorecard for proxy, residual direction, confound, and control probes."),
        ("tables/reward_policy_disagreements.csv", disagreements, "Pairs where proxy, policy, direction, or sycophancy-risk status needs review."),
        ("tables/preference_intervention_results.csv", interventions, "A/B judge-prompt activation-addition results."),
        ("tables/preference_evidence_matrix.csv", evidence, "Method-level evidence matrix for Lab 32."),
    ]
    for rel, rows, desc in specs:
        path = ctx.path(*rel.split("/"))
        bench.write_csv_with_context(ctx, path, rows)
        ctx.register_artifact(path, "table", desc)


def write_state(ctx: bench.RunContext, direction_metadata: Mapping[str, Any], directions: Mapping[str, Mapping[str, Any]]) -> None:
    import torch

    state_path = ctx.path("state", "preference_directions.pt")
    torch.save({name: vec.cpu() for name, vec in directions["steer"].items()}, state_path)
    ctx.register_artifact(state_path, "state", "Steering-scale preference and control directions.")
    meta_path = ctx.path("state", "preference_direction_metadata.json")
    bench.write_json(meta_path, direction_metadata)
    ctx.register_artifact(meta_path, "state", "Direction layer, norm, and scaling metadata.")


def write_method_card(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 32 method card",
        "",
        "This lab audits benign preference signals. It does not claim that a reward proxy understands values.",
        "",
        "- canonical Tier A scorer: DPO-style policy/reference log-prob ratio",
        "- reference baseline: response-unconditional unigram token model",
        "- decode object: residual direction at the prompt+response boundary",
        "- causal test: activation addition on A/B preference prompts",
        "- main null: preference signal is just length, sentiment, politeness, agreement, hedging, or refusal",
        "- forbidden claim: the reward model understands human values",
        "",
        f"- science_ready: `{metrics['science_ready']}`",
        f"- best non-control score: `{metrics['best_non_control_score']}`",
        f"- causal shift over random: `{metrics['causal_shift_over_random']}`",
        "",
        "| method | rung | auc | accuracy | posture |",
        "|---|---|---:|---:|---|",
    ]
    for row in evidence:
        lines.append(f"| {row['method']} | {row['evidence_rung']} | {row['auc']} | {row['accuracy']} | {row['claim_posture']} |")
    path = ctx.path("method_card.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 32 method card and reward/preference boundaries.")


def write_operationalization_audit(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Lab 32 operationalization audit",
        "",
        "Favorite interpretation under attack: a preference or reward direction captures human values.",
        "",
        "## What the measurement can say",
        "",
        "A proxy or residual direction separated preferred from dispreferred benign responses on a frozen pair set, after explicit shortcut controls.",
        "",
        "## What it cannot say",
        "",
        "It cannot say the model has a stable value representation, knows what humans want, or will behave safely under optimization.",
        "",
        "## Cheap explanations",
        "",
        "- Longer answers look better.",
        "- Polite answers look better.",
        "- Agreement with the user looks better.",
        "- Positive sentiment looks better.",
        "- Refusal language looks better on privacy-boundary rows.",
        "- A/B judge prompts may respond to answer-letter priors.",
        "",
        "## Verdicts",
        "",
    ]
    for row in evidence:
        lines.append(f"- `{row['method']}`: `{row['claim_posture']}`.")
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 32 shortcut controls and non-claims.")


def write_safety_status(ctx: bench.RunContext, data_info: Mapping[str, Any]) -> None:
    payload = {
        "lab": LAB_ID,
        "safe_scope": data_info["safety_scope"],
        "blocked_activities": [
            "harmful prompt optimization",
            "jailbreak search",
            "refusal ablation",
            "real private data",
            "real reward-model deployment claims",
        ],
        "data_source": pathlib.Path(str(data_info["data_path"])).name,
        "science_ready": data_info["science_ready"],
    }
    path = ctx.path("diagnostics", "safety_status.json")
    bench.write_json(path, payload)
    ctx.register_artifact(path, "diagnostic", "Safety and scope status for Lab 32.")


def write_run_summary(
    ctx: bench.RunContext,
    data_info: Mapping[str, Any],
    metrics: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    intervention_summary: Mapping[str, Any],
) -> None:
    decode_supported = any(str(r["claim_posture"]) == "beats_confound_controls" for r in evidence)
    causal_supported = any(str(r["claim_posture"]) == "causal_shift_supported" for r in evidence)
    if decode_supported:
        claim = (
            "At least one preference signal beat the shortcut controls on this benign suite. "
            "The claim remains about measured pair preferences, not human values."
        )
    elif causal_supported:
        claim = (
            "No decoded preference signal beat every shortcut control, but activation addition shifted the A/B judge-prompt margin "
            "more than a random direction. This is narrow causal evidence about the judge prompt, not a value claim."
        )
    else:
        claim = (
            "No preference signal cleared every shortcut gate. The useful result is the failure diagnosis: "
            "the proxy or direction is confound-limited under this suite."
        )
    lines = [
        "# Lab 32 run summary: reward models and preference circuits",
        "",
        f"- data rows: {data_info['n_rows_selected']} selected from `{pathlib.Path(str(data_info['data_path'])).name}`",
        f"- domains: `{data_info['domains']}`",
        f"- preference types: `{data_info['preference_types']}`",
        f"- science_ready: `{data_info['science_ready']}`",
        f"- best non-control score: `{metrics['best_non_control_score']}`",
        f"- DPO proxy AUC: `{metrics['dpo_proxy_auc']}`",
        f"- preference direction AUC: `{metrics['preference_direction_auc']}`",
        f"- causal shift over random: `{intervention_summary['causal_shift_over_random']}`",
        "",
        "## Evidence matrix",
        "",
        "| method | rung | auc | accuracy | posture |",
        "|---|---|---:|---:|---|",
    ]
    for row in evidence:
        lines.append(f"| `{row['method']}` | {row['evidence_rung']} | {row['auc']} | {row['accuracy']} | {row['claim_posture']} |")
    lines += [
        "",
        "## Reading order",
        "",
        "1. `method_card.md` for the measurement boundary.",
        "2. `tables/preference_pair_scores.csv` for pair-level proxy and confound margins.",
        "3. `tables/preference_probe_report.csv` for shortcut-control comparisons.",
        "4. `tables/preference_intervention_results.csv` for activation-addition results.",
        "5. `tables/reward_policy_disagreements.csv` before trusting any row-level label.",
        "",
        "## Smallest surviving claim",
        "",
        claim,
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 32 run summary and reading order.")


def write_plot_guide(ctx: bench.RunContext) -> None:
    rows = [
        {"plot": "plots/preference_evidence_dashboard.png", "read_for": "AUC/accuracy of proxy, direction, controls, and intervention shift.", "non_claim": "A preference proxy is not a value model."},
        {"plot": "plots/reward_margin_by_domain.png", "read_for": "DPO proxy preferred margin by domain.", "non_claim": "Domain averages are descriptive."},
        {"plot": "plots/preference_probe_control_atlas.png", "read_for": "Proxy/direction/confound scorecard.", "non_claim": "High AUC can still be shortcut-driven."},
        {"plot": "plots/confound_specificity_ladder.png", "read_for": "Whether preference signals beat shortcut directions.", "non_claim": "Shortcut failure is a valid result."},
        {"plot": "plots/reward_policy_disagreement_matrix.png", "read_for": "Where policy log-prob and DPO proxy disagree with labels.", "non_claim": "Disagreement is not a model preference by itself."},
        {"plot": "plots/preference_steering_frontier.png", "read_for": "Activation-addition dose response.", "non_claim": "Steering a judge prompt is narrow causal evidence."},
        {"plot": "plots/sycophancy_reward_risk_quadrant.png", "read_for": "False-agreement risk rows.", "non_claim": "A quadrant is not a safety evaluation."},
    ]
    path = ctx.path("plots", "plot_reading_guide.csv")
    bench.write_csv(path, rows)
    ctx.register_artifact(path, "table", "Plot reading guide for Lab 32.")


def write_plots(
    ctx: bench.RunContext,
    pairs: Sequence[PreferencePair],
    pair_rows: Sequence[Mapping[str, Any]],
    report: Sequence[Mapping[str, Any]],
    interventions: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
) -> None:
    write_plot_guide(ctx)
    if ctx.args.no_plots:
        return
    import matplotlib.pyplot as plt
    import numpy as np

    report_names = [str(r["score_name"]) for r in report]
    aucs = [float(r["auc"] or 0.0) for r in report]
    accs = [float(r["accuracy"] or 0.0) for r in report]
    top_idx = sorted(range(len(report_names)), key=lambda i: aucs[i], reverse=True)[:10]
    labels = [report_names[i] for i in top_idx]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Lab 32 preference evidence dashboard", fontsize=14, fontweight="bold")
    axes[0, 0].bar(x, [aucs[i] for i in top_idx], color="#0072B2")
    axes[0, 0].axhline(0.5, color="#555555", linestyle="--", linewidth=0.8)
    axes[0, 0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0, 0].set_title("Top score AUCs")
    axes[0, 1].bar(x, [accs[i] for i in top_idx], color="#009E73")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_xticks(x, labels, rotation=35, ha="right")
    axes[0, 1].set_title("Top score accuracies")
    ev_names = [str(r["method"]) for r in evidence]
    axes[1, 0].bar(range(len(ev_names)), [float(r["margin_over_best_confound_auc"] or 0.0) for r in evidence], color="#D55E00")
    axes[1, 0].axhline(0.05, color="#555555", linestyle="--", linewidth=0.8)
    axes[1, 0].set_xticks(range(len(ev_names)), ev_names, rotation=30, ha="right")
    axes[1, 0].set_title("AUC over best confound")
    scale_rows = [r for r in interventions if r["direction"] == "preference_residual_direction"]
    by_scale = defaultdict(list)
    for row in scale_rows:
        by_scale[float(row["scale"])].append(float(row["preferred_answer_logit_margin"]))
    scales = sorted(by_scale)
    axes[1, 1].plot(scales, [safe_mean(by_scale[s], default=0.0) for s in scales], marker="o", color="#CC79A7")
    axes[1, 1].axhline(0, color="#555555", linestyle="--", linewidth=0.8)
    axes[1, 1].set_title("Preference steering frontier")
    axes[1, 1].set_xlabel("scale")
    axes[1, 1].set_ylabel("mean preferred A/B logit margin")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    bench.save_figure(ctx, fig, "preference_evidence_dashboard.png", "Lab 32 preference evidence dashboard.")

    by_domain = defaultdict(list)
    for row, pair in zip(pair_rows, pairs):
        by_domain[pair.domain].append(float(row["dpo_proxy_preferred_margin"] or 0.0))
    domains = sorted(by_domain)
    domain_x = np.arange(len(domains))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(domain_x, [safe_mean(by_domain[d], default=0.0) for d in domains], color="#0072B2")
    ax.axhline(0, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_title("DPO proxy preferred margin by domain")
    ax.set_ylabel("mean preferred margin")
    ax.set_xticks(domain_x, domains, rotation=25, ha="right")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "reward_margin_by_domain.png", "DPO proxy margin by domain.")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(report_names))
    ax.barh(y, aucs, color=["#999999" if "control" in n else "#0072B2" if "dpo" in n or "preference" in n else "#D55E00" for n in report_names])
    ax.axvline(0.5, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_yticks(y, report_names)
    ax.set_xlabel("AUC")
    ax.set_title("Preference probe/control atlas")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "preference_probe_control_atlas.png", "Proxy, direction, confound, and control AUC atlas.")

    ladder = [r for r in report if r["score_family"] in {"proxy", "residual_direction", "confound", "confound_direction"}]
    ladder = sorted(ladder, key=lambda r: float(r["auc"] or 0.0), reverse=True)
    ladder_names = [str(r["score_name"]) for r in ladder]
    ladder_x = np.arange(len(ladder_names))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(ladder_x, [float(r["auc"] or 0.0) for r in ladder], color="#009E73")
    ax.axhline(0.65, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Confound specificity ladder")
    ax.set_ylabel("AUC")
    ax.set_xticks(ladder_x, ladder_names, rotation=35, ha="right")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "confound_specificity_ladder.png", "Preference score compared with shortcut confounds.")

    matrix = np.zeros((2, 2))
    for row, pair in zip(pair_rows, pairs):
        ytrue = 1 if pair.preferred == "a" else 0
        ypred = 1 if str(row["dpo_proxy_prediction"]) == "a" else 0
        matrix[1 - ytrue, ypred] += 1
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["pred B", "pred A"])
    ax.set_yticks([0, 1], ["true A", "true B"])
    ax.set_title("DPO proxy disagreement matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="#111111")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "reward_policy_disagreement_matrix.png", "DPO proxy predicted-vs-label matrix.")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for direction, color in (("preference_residual_direction", "#CC79A7"), ("random_direction_control", "#999999")):
        by_scale = defaultdict(list)
        for row in interventions:
            if row["direction"] == direction:
                by_scale[float(row["scale"])].append(float(row["preferred_answer_logit_margin"]))
        xs = sorted(by_scale)
        ax.plot(xs, [safe_mean(by_scale[s], default=0.0) for s in xs], marker="o", label=direction, color=color)
    ax.axhline(0, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_xlabel("activation-addition scale")
    ax.set_ylabel("mean preferred A/B logit margin")
    ax.set_title("Preference steering frontier")
    ax.legend(fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "preference_steering_frontier.png", "Activation-addition preference frontier.")

    syco_rows = [(pair, row) for pair, row in zip(pairs, pair_rows) if pair.preference_type == "anti_sycophancy"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if syco_rows:
        xs = [float(row["agreement_preferred_margin"] or 0.0) for _, row in syco_rows]
        ys = [float(row["dpo_proxy_preferred_margin"] or 0.0) for _, row in syco_rows]
        colors = ["#009E73" if float(row["dpo_proxy_preferred_margin"] or 0.0) > 0 else "#D55E00" for _, row in syco_rows]
        ax.scatter(xs, ys, c=colors, s=80)
        for pair, row in syco_rows:
            ax.annotate(pair.pair_id.replace("sycophancy_", "syco_"), (float(row["agreement_preferred_margin"] or 0.0), float(row["dpo_proxy_preferred_margin"] or 0.0)), fontsize=8)
    ax.axhline(0, color="#555555", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_xlabel("agreement preferred margin")
    ax.set_ylabel("DPO proxy preferred margin")
    ax.set_title("Sycophancy reward-risk quadrant")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "sycophancy_reward_risk_quadrant.png", "Sycophancy risk quadrant for false-agreement rows.")


def write_claims(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> None:
    run_name = ctx.run_dir.name
    claims = []
    for i, row in enumerate(evidence, start=1):
        claims.append({
            "id": f"{LAB_ID}-C{i}",
            "tag": str(row["evidence_rung"]).replace("_CONTROL", ""),
            "text": (
                f"Method `{row['method']}` reached AUC {row['auc']} and accuracy {row['accuracy']} on "
                f"{metrics['n_pairs']} benign preference pairs; posture `{row['claim_posture']}`."
            ),
            "artifact": f"runs/{run_name}/tables/preference_evidence_matrix.csv",
            "falsifier": "Length, sentiment, agreement, refusal, random-direction, or held-out disagreement controls explain the measured margin.",
        })
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    pairs, data_info = load_pairs(ctx)
    manifest_path = ctx.path("diagnostics", "data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Lab 32 data manifest and preference scope.")
    write_safety_status(ctx, data_info)
    bench.run_hook_parity_check(ctx, bundle, pairs[0].prompt)
    first = bench.run_with_residual_cache(bundle, pairs[0].prompt)
    bench.run_lens_self_check(ctx, bundle, first)
    bench.run_patch_noop_check(ctx, bundle, pairs[0].prompt)

    steer_layer, stream_depth = choose_layers(bundle)
    pair_rows = score_pairs(bundle, pairs)
    vectors = capture_response_vectors(bundle, pairs, stream_depth)
    directions, direction_metadata = build_directions(ctx, bundle, pairs, pair_rows, vectors)
    margins = direction_margins(pairs, vectors, directions["unit"])
    attach_direction_scores(pair_rows, pairs, margins)
    add_shuffled_control(pair_rows)
    specs = score_specs() + [("shuffled_score_control", "shuffled_score_control_margin_a_minus_b", "control")]
    report = summarize_scores(pairs, pair_rows, specs)
    interventions, intervention_summary = run_interventions(
        bundle,
        pairs,
        steer_layer,
        directions["steer"]["preference_residual_direction"],
        directions["steer"]["random_direction_control"],
    )
    disagreements = build_disagreements(pairs, pair_rows)
    evidence, metrics = build_evidence(report, intervention_summary, data_info)
    metrics = {**metrics, "direction_metadata": direction_metadata, "intervention": intervention_summary, "data": data_info}

    write_tables(ctx, pair_rows, report, disagreements, interventions, evidence)
    write_state(ctx, direction_metadata, directions)
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 32 metrics.")
    write_method_card(ctx, evidence, metrics)
    write_operationalization_audit(ctx, evidence)
    write_run_summary(ctx, data_info, metrics, evidence, intervention_summary)
    write_claims(ctx, evidence, metrics)
    write_plots(ctx, pairs, pair_rows, report, interventions, evidence)
    print(f"[lab32] wrote {len(pair_rows)} pair scores, {len(report)} probe rows, and {len(evidence)} evidence rows")
