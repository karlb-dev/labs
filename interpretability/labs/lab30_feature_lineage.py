"""Lab 30: Feature lineage without feature identity.

This lab builds supervised prototype directions at multiple residual depths and
asks whether those directions form a recurring lineage across layers. It is a
conservative first pass: no SAE, no transcoder, no crosscoder, and no external
cross-model comparison. The output is allowed to talk about recurring supervised
prototype-direction handles. It is not allowed to talk about feature identity.

Evidence levels:
  * DECODE: held-out domain labels are decodable from the direction.
  * ATTR: adjacent-depth direction/activation/top-context evidence supports a
    lineage edge above random and confusable controls.
  * CAUSAL, narrow: activation addition moves a marker-token margin. This is a
    side-channel probe, not semantic steering.
  * AUDIT: counterexamples, confusable controls, split/merge screens, and the
    external-cross-model non-result are kept as first-class artifacts.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import pathlib
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import interp_bench as bench

LAB_ID = "L30"
LAB_NAME = "lab30_feature_lineage"
DATA_FILE = "feature_lineage_corpus.csv"

# Prompt-set caps are per domain. The bench's global --max-examples is applied
# afterward as a balanced total cap.
PROMPT_SET_DOMAIN_CAPS = {"small": 4, "medium": 6, "full": 0}
EVAL_SPLITS = {"eval", "heldout", "test"}
TRAIN_SPLITS = {"train", "discovery"}

LINEAGE_PASS_SCORE = 0.62
LINEAGE_PASS_LIFT = 0.08
CONFUSABLE_PASS_GAP = 0.03
NODE_AUC_PASS = 0.70
NODE_RANDOM_LIFT_PASS = 0.05
TRANSFER_SCALE_FRACTION = 0.45
TOP_CONTEXTS_K = 5
MIN_DOMAIN_ROWS = 3
RESIDUAL_ADD_NOOP_ATOL = 1e-4

REQUIRED_COLUMNS = {
    "row_id",
    "family",
    "domain",
    "source_lab",
    "text",
    "group_id",
    "split",
    "labels_json",
}
REQUIRED_LABEL_KEYS = {"marker_token", "contrast_token", "confusable_domain"}


@dataclasses.dataclass
class CorpusRow:
    row_id: str
    family: str
    domain: str
    source_lab: str
    text: str
    group_id: str
    split: str
    labels: dict[str, Any]
    input_ids: list[int] = dataclasses.field(default_factory=list)
    marker_id: int = -1
    contrast_id: int = -1
    n_tokens: int = 0


# ---------------------------------------------------------------------------
# Numeric and file helpers
# ---------------------------------------------------------------------------


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=bench.json_default).encode("utf-8")).hexdigest()


def rounded(value: Any, digits: int = 4) -> Any:
    try:
        f = float(value)
    except Exception:
        return value
    return round(f, digits) if math.isfinite(f) else ""


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        f = float(value)
    except Exception:
        return default
    return f if math.isfinite(f) else default


def safe_mean(values: Sequence[Any], default: float = float("nan")) -> float:
    vals = [as_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(statistics.fmean(vals)) if vals else default


def safe_stdev(values: Sequence[Any], default: float = float("nan")) -> float:
    vals = [as_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if len(vals) < 2:
        return default
    return float(statistics.stdev(vals))


def safe_max(values: Sequence[Any], default: float = float("nan")) -> float:
    vals = [as_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return max(vals) if vals else default


def safe_corr(xs: Sequence[Any], ys: Sequence[Any]) -> float:
    pairs: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        xf = as_float(x)
        yf = as_float(y)
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
    pairs = []
    for y, s in zip(labels, scores):
        sf = as_float(s)
        if math.isfinite(sf):
            pairs.append((int(y), sf))
    pos = [s for y, s in pairs if y == 1]
    neg = [s for y, s in pairs if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for ps in pos:
        for ns in neg:
            wins += 1.0 if ps > ns else 0.5 if ps == ns else 0.0
    return wins / (len(pos) * len(neg))


def cosine(a: Any, b: Any) -> float:
    denom = a.float().norm() * b.float().norm()
    if float(denom) <= 1e-12:
        return float("nan")
    return float((a.float() @ b.float()) / denom)


def unit_vector(vector: Any) -> Any:
    norm = vector.float().norm().clamp_min(1e-8)
    return vector.float() / norm


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(a)
    sb = set(b)
    return len(sa & sb) / len(sa | sb) if sa or sb else 0.0


def split_group(row: CorpusRow) -> str:
    s = row.split.strip().lower()
    if s in EVAL_SPLITS:
        return "eval"
    if s in TRAIN_SPLITS:
        return "train"
    return s or "unspecified"


def is_train(row: CorpusRow) -> bool:
    return split_group(row) == "train"


def is_eval(row: CorpusRow) -> bool:
    return split_group(row) == "eval"


def row_confusable(row: CorpusRow) -> str:
    return str(row.labels.get("confusable_domain", ""))


def row_marker(row: CorpusRow) -> str:
    return str(row.labels.get("marker_token", ""))


def row_contrast(row: CorpusRow) -> str:
    return str(row.labels.get("contrast_token", ""))


def write_jsonl(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), sort_keys=True, default=bench.json_default) + "\n")


# ---------------------------------------------------------------------------
# Built-in smoke fallback and data loading
# ---------------------------------------------------------------------------


def builtin_smoke_rows() -> list[dict[str, str]]:
    """Small fallback only for Tier A if the frozen CSV is absent."""
    examples = {
        "code": [
            "The code test checked the parser and the error path.",
            "A Python loop sorted the list and returned a stable result.",
            "The repository diff changed the tokenizer wrapper.",
            "The stack trace pointed to the cache bug.",
        ],
        "cooking": [
            "The recipe mixed herbs into the sauce after the onions softened.",
            "A chef whisked batter until the bowl looked smooth.",
            "The soup simmered while carrots became tender.",
            "The bread crust turned golden near the oven door.",
        ],
        "finance": [
            "The bank reviewed the loan and the repayment history.",
            "The fund compared bond yield with inflation forecasts.",
            "The budget tracked payroll, invoices, taxes, and cash.",
            "The exchange rate moved after the central bank statement.",
        ],
        "sports": [
            "The coach reviewed match film and the defensive formation.",
            "A striker scored late after the goalkeeper missed the ball.",
            "The team practiced passing lanes near midfield.",
            "The scoreboard changed after a fast break.",
        ],
    }
    meta = {
        "code": ("technical_procedure", "lab08_sae_transcoders", "code_cooking", " code", " food", "cooking"),
        "cooking": ("household_procedure", "lab08_sae_transcoders", "code_cooking", " food", " code", "code"),
        "finance": ("competitive_numbers", "lab19_model_diffing_crosscoders", "finance_sports", " money", " game", "sports"),
        "sports": ("competitive_numbers", "lab19_model_diffing_crosscoders", "finance_sports", " game", " money", "finance"),
    }
    out: list[dict[str, str]] = []
    for domain, texts in examples.items():
        family, source_lab, group_id, marker, contrast, confusable = meta[domain]
        for i, text in enumerate(texts):
            labels = {
                "domain": domain,
                "confusable_domain": confusable,
                "marker_token": marker,
                "contrast_token": contrast,
                "neutral_probe_prompt": "This passage is about",
                "surface_family": family,
            }
            out.append({
                "row_id": f"smoke_{domain}_{i}",
                "family": family,
                "domain": domain,
                "source_lab": source_lab,
                "text": text,
                "group_id": group_id,
                "split": "eval" if i == 2 else "train",
                "labels_json": json.dumps(labels, sort_keys=True),
            })
    return out


def data_path(args: Any) -> pathlib.Path:
    prompt_set = str(getattr(args, "prompt_set", "") or "")
    candidate = pathlib.Path(prompt_set)
    if prompt_set not in PROMPT_SET_DOMAIN_CAPS and candidate.suffix.lower() in {".csv", ".tsv"}:
        return candidate if candidate.is_absolute() else (bench.COURSE_ROOT / candidate).resolve()
    return bench.COURSE_ROOT / "data" / DATA_FILE


def parse_row(raw: Mapping[str, str]) -> CorpusRow:
    row_id = str(raw["row_id"]).strip()
    try:
        labels = json.loads(raw["labels_json"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{row_id}: labels_json is invalid JSON: {exc}") from exc
    if not isinstance(labels, dict):
        raise ValueError(f"{row_id}: labels_json must decode to an object")
    missing = sorted(REQUIRED_LABEL_KEYS - set(labels))
    if missing:
        raise ValueError(f"{row_id}: labels_json missing keys {missing}")
    return CorpusRow(
        row_id=row_id,
        family=str(raw["family"]).strip(),
        domain=str(raw["domain"]).strip(),
        source_lab=str(raw["source_lab"]).strip(),
        text=str(raw["text"]).strip(),
        group_id=str(raw["group_id"]).strip(),
        split=str(raw["split"]).strip().lower(),
        labels=dict(labels),
    )


def apply_caps(rows: list[CorpusRow], args: Any) -> list[CorpusRow]:
    prompt_set = str(getattr(args, "prompt_set", "") or "small")
    per_domain_cap = PROMPT_SET_DOMAIN_CAPS.get(prompt_set, 0)
    by_domain: dict[str, list[CorpusRow]] = defaultdict(list)
    for row in rows:
        by_domain[row.domain].append(row)
    selected: list[CorpusRow] = []
    for domain in sorted(by_domain):
        domain_rows = by_domain[domain]
        if not per_domain_cap:
            selected.extend(domain_rows)
            continue
        # Preserve split coverage under small/medium caps. The frozen CSV is
        # train-first inside each domain, so a naive first-N cap would create
        # train-only smoke runs and counterfeit the split-generalization audit.
        buckets = {
            "train": [row for row in domain_rows if split_group(row) == "train"],
            "eval": [row for row in domain_rows if split_group(row) == "eval"],
            "other": [row for row in domain_rows if split_group(row) not in {"train", "eval"}],
        }
        if per_domain_cap <= 4:
            quotas = {"train": max(1, per_domain_cap - 2), "eval": 1, "other": 1}
        elif per_domain_cap <= 6:
            quotas = {"train": 3, "eval": 2, "other": 1}
        else:
            quotas = {"train": max(2, per_domain_cap // 2), "eval": max(1, per_domain_cap // 3), "other": per_domain_cap}
        picked: list[CorpusRow] = []
        for split in ("train", "eval", "other"):
            picked.extend(buckets[split][: quotas.get(split, 0)])
        if len(picked) < per_domain_cap:
            seen = {row.row_id for row in picked}
            for row in domain_rows:
                if row.row_id not in seen:
                    picked.append(row)
                    seen.add(row.row_id)
                    if len(picked) >= per_domain_cap:
                        break
        selected.extend(picked[:per_domain_cap])

    max_examples = int(getattr(args, "max_examples", 0) or 0)
    if max_examples > 0 and len(selected) > max_examples:
        by_domain = defaultdict(list)
        for row in selected:
            by_domain[row.domain].append(row)
        balanced: list[CorpusRow] = []
        cursor = 0
        domains = sorted(by_domain)
        while len(balanced) < max_examples:
            progressed = False
            for domain in domains:
                if cursor < len(by_domain[domain]):
                    balanced.append(by_domain[domain][cursor])
                    progressed = True
                    if len(balanced) >= max_examples:
                        break
            if not progressed:
                break
            cursor += 1
        selected = balanced
    return selected


def load_rows(ctx: bench.RunContext) -> tuple[list[CorpusRow], dict[str, Any]]:
    path = data_path(ctx.args)
    data_source = "frozen_csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            raw_rows = [dict(row) for row in csv.DictReader(f)]
        sha = file_sha256(path)
    else:
        if str(getattr(ctx.args, "tier", "")).lower() != "a":
            raise FileNotFoundError(f"Lab 30 data file not found: {path}")
        print("[lab30] data CSV missing; using built-in Tier A smoke fallback. Do not ledger claims from this run.")
        raw_rows = builtin_smoke_rows()
        data_source = "builtin_tier_a_smoke_fallback"
        sha = stable_json_sha(raw_rows)

    if raw_rows:
        missing = sorted(REQUIRED_COLUMNS - set(raw_rows[0]))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
    all_rows = [parse_row(row) for row in raw_rows]
    selected = apply_caps(all_rows, ctx.args)
    domains = sorted({row.domain for row in selected})
    groups = sorted({row.group_id for row in selected})
    info = {
        "data_source": data_source,
        "science_ready_data": data_source == "frozen_csv",
        "data_path": str(path),
        "data_sha256": sha,
        "n_rows_file": len(all_rows),
        "n_rows_selected": len(selected),
        "prompt_set": ctx.args.prompt_set,
        "max_examples": ctx.args.max_examples,
        "domains": {d: sum(1 for row in selected if row.domain == d) for d in domains},
        "groups": {g: sum(1 for row in selected if row.group_id == g) for g in groups},
        "splits": dict(Counter(split_group(row) for row in selected)),
        "science_scope": "supervised prototype directions for cross-layer lineage; no SAE/crosscoder/external-cross-model run",
    }
    if not selected:
        raise RuntimeError("Lab 30 selected zero rows.")
    return selected, info


# ---------------------------------------------------------------------------
# Validation, tokenization, and baseline capture
# ---------------------------------------------------------------------------


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def tokenization_gate(ctx: bench.RunContext, bundle: bench.ModelBundle, rows: list[CorpusRow]) -> tuple[list[CorpusRow], list[dict[str, Any]]]:
    tok = bundle.tokenizer
    kept: list[CorpusRow] = []
    audit: list[dict[str, Any]] = []
    for row in rows:
        problems: list[str] = []
        warnings: list[str] = []
        encoded = tok(row.text, add_special_tokens=True)["input_ids"]
        marker_ids = token_ids(tok, row_marker(row))
        contrast_ids = token_ids(tok, row_contrast(row))
        if not encoded:
            problems.append("empty_text")
        if len(marker_ids) != 1:
            problems.append(f"marker_token_count={len(marker_ids)}")
        if len(contrast_ids) != 1:
            problems.append(f"contrast_token_count={len(contrast_ids)}")
        if len(marker_ids) == 1 and len(contrast_ids) == 1 and marker_ids[0] == contrast_ids[0]:
            problems.append("marker_equals_contrast")
        if row.domain == row_confusable(row):
            problems.append("confusable_domain_equals_domain")
        if split_group(row) not in {"train", "eval"}:
            warnings.append(f"nonstandard_split={row.split}")
        if not problems:
            row.input_ids = list(encoded)
            row.marker_id = int(marker_ids[0])
            row.contrast_id = int(contrast_ids[0])
            row.n_tokens = len(encoded)
            kept.append(row)
        audit.append({
            "row_id": row.row_id,
            "domain": row.domain,
            "confusable_domain": row_confusable(row),
            "group_id": row.group_id,
            "split": split_group(row),
            "n_tokens": len(encoded),
            "marker_token": row_marker(row),
            "marker_token_count": len(marker_ids),
            "marker_id": marker_ids[0] if len(marker_ids) == 1 else "",
            "contrast_token": row_contrast(row),
            "contrast_token_count": len(contrast_ids),
            "contrast_id": contrast_ids[0] if len(contrast_ids) == 1 else "",
            "kept": not problems,
            "problems": ";".join(problems),
            "warnings": ";".join(warnings),
            "text": row.text,
        })
    path = ctx.path("diagnostics", "tokenization_gate.csv")
    bench.write_csv_with_context(ctx, path, audit)
    ctx.register_artifact(path, "diagnostic", "Runtime text and marker/contrast token audit for Lab 30.")
    if not kept:
        raise RuntimeError("Lab 30 tokenization gate dropped every row.")
    print(f"[lab30] tokenization gate kept {len(kept)}/{len(rows)} rows")
    return kept, audit


def corpus_manifest(ctx: bench.RunContext, rows: Sequence[CorpusRow]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for domain in sorted({row.domain for row in rows}):
        drows = [row for row in rows if row.domain == domain]
        conf = sorted({row_confusable(row) for row in drows})
        out.append({
            "domain": domain,
            "confusable_domains": ";".join(conf),
            "group_ids": ";".join(sorted({row.group_id for row in drows})),
            "families": ";".join(sorted({row.family for row in drows})),
            "source_labs": ";".join(sorted({row.source_lab for row in drows})),
            "n_rows": len(drows),
            "n_train": sum(1 for row in drows if is_train(row)),
            "n_eval": sum(1 for row in drows if is_eval(row)),
            "n_marker_tokens": len({row.marker_id for row in drows if row.marker_id >= 0}),
            "n_contrast_tokens": len({row.contrast_id for row in drows if row.contrast_id >= 0}),
            "science_ready_domain": len(drows) >= MIN_DOMAIN_ROWS and any(is_train(row) for row in drows) and any(is_eval(row) for row in drows),
        })
    path = ctx.path("tables", "corpus_manifest.csv")
    bench.write_csv_with_context(ctx, path, out)
    ctx.register_artifact(path, "table", "Domain, confusable-group, and split audit for the Lab 30 corpus.")
    split_path = ctx.path("diagnostics", "split_balance.csv")
    bench.write_csv_with_context(ctx, split_path, out)
    ctx.register_artifact(split_path, "diagnostic", "Domain and split coverage audit for Lab 30 after tokenization.")
    return out


def coarse_depths(n_layers: int, prompt_set: str) -> list[int]:
    if prompt_set == "full":
        return list(range(n_layers + 1))
    fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
    return sorted({max(0, min(n_layers, int(round(n_layers * f)))) for f in fractions})


def depth_claimable(bundle: bench.ModelBundle, depth: int) -> bool:
    return 0 < int(depth) < bundle.anatomy.n_layers


def capture_corpus(ctx: bench.RunContext, bundle: bench.ModelBundle, rows: Sequence[CorpusRow]) -> tuple[dict[str, Any], list[int], dict[int, float]]:
    captures: dict[str, Any] = {}
    depths = coarse_depths(bundle.anatomy.n_layers, str(ctx.args.prompt_set))
    report_every = max(1, len(rows) // 4)
    for i, row in enumerate(rows, start=1):
        cap = bench.run_with_residual_cache(bundle, row.text)
        captures[row.row_id] = cap
        row.input_ids = list(cap.input_ids)
        row.n_tokens = len(cap.input_ids)
        if i % report_every == 0 or i == len(rows):
            print(f"[lab30] captured {i}/{len(rows)} corpus rows")
    norm_by_depth: dict[int, float] = {}
    norm_rows: list[dict[str, Any]] = []
    for depth in depths:
        vals = []
        for row in rows:
            norm = float(captures[row.row_id].streams[depth, -1].float().norm())
            vals.append(norm)
            norm_rows.append({
                "row_id": row.row_id,
                "domain": row.domain,
                "split": split_group(row),
                "depth": depth,
                "claimable_depth": depth_claimable(bundle, depth),
                "final_token_residual_norm": rounded(norm),
            })
        norm_by_depth[depth] = statistics.median(vals) if vals else 1.0
    norm_path = ctx.path("diagnostics", "activation_norms_by_depth.csv")
    bench.write_csv_with_context(ctx, norm_path, norm_rows)
    ctx.register_artifact(norm_path, "diagnostic", "Final-token residual norms by selected depth for Lab 30.")
    return captures, depths, norm_by_depth


# ---------------------------------------------------------------------------
# Directions, nodes, and edges
# ---------------------------------------------------------------------------


def rows_for_fit(rows: Sequence[CorpusRow]) -> list[CorpusRow]:
    train = [row for row in rows if is_train(row)]
    return train if train else list(rows)


def eval_rows(rows: Sequence[CorpusRow]) -> list[CorpusRow]:
    ev = [row for row in rows if is_eval(row)]
    return ev if ev else list(rows)


def deterministic_random_like(vector: Any, key: str) -> Any:
    import torch

    gen = torch.Generator(device="cpu").manual_seed(stable_int(key) % (2**31 - 1))
    rand = torch.randn(vector.shape, generator=gen, dtype=vector.float().dtype)
    return unit_vector(rand) * vector.float().norm().clamp_min(1e-8)


def activation(captures: Mapping[str, Any], row: CorpusRow, depth: int) -> Any:
    return captures[row.row_id].streams[depth, -1].float().cpu()


def raw_direction_for_domain(rows: Sequence[CorpusRow], captures: Mapping[str, Any], domain: str, depth: int) -> Any:
    import torch

    pos = [activation(captures, row, depth) for row in rows if row.domain == domain]
    neg = [activation(captures, row, depth) for row in rows if row.domain != domain]
    if not pos or not neg:
        raise ValueError(f"Cannot build direction for domain {domain} at depth {depth}: pos={len(pos)} neg={len(neg)}")
    return torch.stack(pos).mean(dim=0) - torch.stack(neg).mean(dim=0)


def score_direction(captures: Mapping[str, Any], row: CorpusRow, depth: int, direction: Any) -> float:
    return float(activation(captures, row, depth).dot(direction.float().cpu()))


def orient_direction_on_train(rows: Sequence[CorpusRow], captures: Mapping[str, Any], domain: str, depth: int, direction: Any) -> Any:
    train = rows_for_fit(rows)
    pos = [score_direction(captures, row, depth, direction) for row in train if row.domain == domain]
    neg = [score_direction(captures, row, depth, direction) for row in train if row.domain != domain]
    if pos and neg and safe_mean(pos, 0.0) < safe_mean(neg, 0.0):
        return -direction
    return direction


def auc_for_rows(rows: Sequence[CorpusRow], captures: Mapping[str, Any], domain: str, depth: int, direction: Any) -> float:
    labels = [1 if row.domain == domain else 0 for row in rows]
    scores = [score_direction(captures, row, depth, direction) for row in rows]
    return auc_binary(labels, scores)


def top_contexts(rows: Sequence[CorpusRow], scores: Mapping[str, float], k: int = TOP_CONTEXTS_K) -> list[str]:
    ranked = sorted(rows, key=lambda row: scores.get(row.row_id, float("-inf")), reverse=True)
    return [row.row_id for row in ranked[:k]]


def build_nodes(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    rows: Sequence[CorpusRow],
    captures: Mapping[str, Any],
    depths: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    domains = sorted({row.domain for row in rows})
    fit = rows_for_fit(rows)
    ev = eval_rows(rows)
    nodes: list[dict[str, Any]] = []
    state: dict[tuple[str, int], dict[str, Any]] = {}
    for depth in depths:
        for domain in domains:
            raw_direction = raw_direction_for_domain(fit, captures, domain, depth)
            raw_norm = float(raw_direction.float().norm())
            direction = orient_direction_on_train(rows, captures, domain, depth, unit_vector(raw_direction))
            random_direction = deterministic_random_like(direction, f"node|{domain}|{depth}")
            random_direction = orient_direction_on_train(rows, captures, domain, depth, unit_vector(random_direction))
            all_scores = {row.row_id: score_direction(captures, row, depth, direction) for row in rows}
            train_scores = {row.row_id: all_scores[row.row_id] for row in fit}
            eval_scores = {row.row_id: all_scores[row.row_id] for row in ev}
            random_all_scores = {row.row_id: score_direction(captures, row, depth, random_direction) for row in rows}
            train_auc = auc_for_rows(fit, captures, domain, depth, direction)
            eval_auc = auc_for_rows(ev, captures, domain, depth, direction)
            random_train_auc = auc_for_rows(fit, captures, domain, depth, random_direction)
            random_eval_auc = auc_for_rows(ev, captures, domain, depth, random_direction)
            top_train = top_contexts(fit, train_scores)
            top_eval = top_contexts(ev, eval_scores)
            node_id = f"{domain}@d{depth}"
            claimable = depth_claimable(bundle, depth) and math.isfinite(eval_auc) and (eval_auc >= NODE_AUC_PASS) and ((eval_auc - random_eval_auc) >= NODE_RANDOM_LIFT_PASS)
            nodes.append({
                "node_id": node_id,
                "model": bundle.anatomy.model_id,
                "domain": domain,
                "confusable_domain": sorted({row_confusable(row) for row in rows if row.domain == domain})[0],
                "group_id": sorted({row.group_id for row in rows if row.domain == domain})[0],
                "depth": depth,
                "claimable_depth": depth_claimable(bundle, depth),
                "feature_kind": "supervised_prototype_direction",
                "direction_norm_raw": rounded(raw_norm),
                "direction_norm_used": rounded(float(direction.float().norm())),
                "train_auc": rounded(train_auc),
                "eval_auc": rounded(eval_auc),
                "random_train_auc": rounded(random_train_auc),
                "random_eval_auc": rounded(random_eval_auc),
                "eval_auc_lift_over_random": rounded(eval_auc - random_eval_auc if math.isfinite(eval_auc) and math.isfinite(random_eval_auc) else float("nan")),
                "claimable_node": claimable,
                "top_train_contexts": " ".join(top_train),
                "top_eval_contexts": " ".join(top_eval),
                "n_positive_rows": sum(1 for row in rows if row.domain == domain),
                "n_train_rows": sum(1 for row in fit if row.domain == domain),
                "n_eval_rows": sum(1 for row in ev if row.domain == domain),
                "n_rows": len(rows),
            })
            state[(domain, depth)] = {
                "direction": direction.float().cpu(),
                "raw_direction": raw_direction.float().cpu(),
                "random_direction": random_direction.float().cpu(),
                "scores": all_scores,
                "random_scores": random_all_scores,
                "top_train_contexts": top_train,
                "top_eval_contexts": top_eval,
                "train_auc": train_auc,
                "eval_auc": eval_auc,
                "random_train_auc": random_train_auc,
                "random_eval_auc": random_eval_auc,
                "claimable_node": claimable,
            }
    path = ctx.path("tables", "feature_lineage_nodes.csv")
    bench.write_csv_with_context(ctx, path, nodes)
    ctx.register_artifact(path, "table", "Layerwise supervised prototype-direction nodes with held-out and random-control AUCs.")
    return nodes, state


def lineage_score(cos_value: float, corr_value: float, top_jaccard: float, source_auc: float, target_auc: float) -> float:
    cos_part = (cos_value + 1.0) / 2.0 if math.isfinite(cos_value) else 0.0
    corr_part = (corr_value + 1.0) / 2.0 if math.isfinite(corr_value) else 0.0
    auc_part = max(0.0, min(1.0, safe_mean([source_auc, target_auc], default=0.0)))
    return 0.30 * cos_part + 0.30 * corr_part + 0.20 * top_jaccard + 0.20 * auc_part


def edge_score_from_state(
    rows: Sequence[CorpusRow],
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    use_random_source: bool = False,
) -> tuple[float, float, float, float]:
    ev = eval_rows(rows)
    source_scores_key = "random_scores" if use_random_source else "scores"
    source_scores = [source[source_scores_key][row.row_id] for row in ev]
    target_scores = [target["scores"][row.row_id] for row in ev]
    corr = safe_corr(source_scores, target_scores)
    top_j = 0.0 if use_random_source else jaccard(source["top_eval_contexts"], target["top_eval_contexts"])
    cos_val = cosine(source["random_direction" if use_random_source else "direction"], target["direction"])
    auc_source = float(source["random_eval_auc"] if use_random_source else source["eval_auc"])
    auc_target = float(target["eval_auc"])
    return lineage_score(cos_val, corr, top_j, auc_source, auc_target), cos_val, corr, top_j


def build_edges(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    rows: Sequence[CorpusRow],
    depths: Sequence[int],
    node_state: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    domains = sorted({row.domain for row in rows})
    confusable_by_domain = {domain: sorted({row_confusable(row) for row in rows if row.domain == domain})[0] for domain in domains}
    raw_edges: list[dict[str, Any]] = []
    score_lookup: dict[tuple[str, str, int, int], float] = {}
    for d0, d1 in zip(depths, depths[1:]):
        for source_domain in domains:
            source = node_state[(source_domain, d0)]
            for target_domain in domains:
                target = node_state[(target_domain, d1)]
                score, cos_value, corr_value, top_j = edge_score_from_state(rows, source, target)
                rand_score, rand_cos, rand_corr, rand_top_j = edge_score_from_state(rows, source, target, use_random_source=True)
                key = (source_domain, target_domain, int(d0), int(d1))
                score_lookup[key] = score
                source_claimable = bool(source["claimable_node"])
                target_claimable = bool(target["claimable_node"])
                claimable_edge = depth_claimable(bundle, d0) and depth_claimable(bundle, d1) and source_claimable and target_claimable
                raw_edges.append({
                    "edge_id": f"{source_domain}@d{d0}->{target_domain}@d{d1}",
                    "source_node": f"{source_domain}@d{d0}",
                    "target_node": f"{target_domain}@d{d1}",
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "confusable_target_domain": confusable_by_domain.get(source_domain, ""),
                    "source_depth": d0,
                    "target_depth": d1,
                    "same_label": source_domain == target_domain,
                    "is_confusable_edge": target_domain == confusable_by_domain.get(source_domain, ""),
                    "claimable_edge": claimable_edge,
                    "decoder_cosine_proxy": rounded(cos_value),
                    "activation_correlation_eval": rounded(corr_value),
                    "top_context_jaccard_eval": rounded(top_j),
                    "source_eval_auc": rounded(source["eval_auc"]),
                    "target_eval_auc": rounded(target["eval_auc"]),
                    "source_random_eval_auc": rounded(source["random_eval_auc"]),
                    "lineage_score": rounded(score),
                    "random_control_score": rounded(rand_score),
                    "random_control_cosine": rounded(rand_cos),
                    "random_control_correlation_eval": rounded(rand_corr),
                    "lineage_lift_over_random": rounded(score - rand_score),
                })
    for row in raw_edges:
        conf_target = row["confusable_target_domain"]
        conf_score = score_lookup.get((row["source_domain"], conf_target, int(row["source_depth"]), int(row["target_depth"])), float("nan"))
        strongest_comp = safe_max([
            score for (sd, td, d0, d1), score in score_lookup.items()
            if sd == row["source_domain"] and d0 == int(row["source_depth"]) and d1 == int(row["target_depth"]) and td != row["source_domain"]
        ])
        score = as_float(row["lineage_score"])
        rand = as_float(row["random_control_score"])
        random_gap = score - rand if math.isfinite(score) and math.isfinite(rand) else float("nan")
        conf_gap = score - conf_score if math.isfinite(score) and math.isfinite(conf_score) else float("nan")
        row["confusable_control_score"] = rounded(conf_score)
        row["confusable_gap"] = rounded(conf_gap)
        row["strongest_nonself_score"] = rounded(strongest_comp)
        row["claim_candidate"] = bool(
            row["same_label"]
            and row["claimable_edge"]
            and score >= LINEAGE_PASS_SCORE
            and random_gap >= LINEAGE_PASS_LIFT
            and conf_gap >= CONFUSABLE_PASS_GAP
        )
        row["failed_gate"] = "" if row["claim_candidate"] else failure_gate(row)
    path = ctx.path("tables", "feature_lineage_edges.csv")
    bench.write_csv_with_context(ctx, path, raw_edges)
    ctx.register_artifact(path, "table", "Adjacent-depth lineage edges with random and confusable controls.")
    return raw_edges


def failure_gate(edge: Mapping[str, Any]) -> str:
    if not edge.get("same_label"):
        return "not_same_label_edge"
    if not edge.get("claimable_edge"):
        return "endpoint_node_or_depth_not_claimable"
    if as_float(edge.get("lineage_score")) < LINEAGE_PASS_SCORE:
        return "lineage_score_below_bar"
    if as_float(edge.get("lineage_lift_over_random")) < LINEAGE_PASS_LIFT:
        return "random_control_too_close"
    if as_float(edge.get("confusable_gap")) < CONFUSABLE_PASS_GAP:
        return "confusable_edge_too_close"
    return "unknown"


# ---------------------------------------------------------------------------
# Split/merge, transfer, evidence, and state
# ---------------------------------------------------------------------------


def split_merge_tables(ctx: bench.RunContext, edge_rows: Sequence[Mapping[str, Any]], depths: Sequence[int], domains: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    claimable = [row for row in edge_rows if row.get("claimable_edge")]
    source_depths = sorted({int(row["source_depth"]) for row in claimable}) or list(depths[:-1])
    for d0 in source_depths:
        d1_candidates = sorted({int(row["target_depth"]) for row in edge_rows if int(row["source_depth"]) == d0})
        if not d1_candidates:
            continue
        d1 = d1_candidates[0]
        for source_domain in domains:
            outgoing = [row for row in edge_rows if row["source_domain"] == source_domain and int(row["source_depth"]) == d0 and int(row["target_depth"]) == d1]
            top = sorted(outgoing, key=lambda row: as_float(row.get("lineage_score"), -999.0), reverse=True)[:3]
            scores = [max(1e-8, as_float(row.get("lineage_score"), 0.0)) for row in top]
            total = sum(scores)
            entropy = -sum((s / total) * math.log(s / total) for s in scores) / math.log(len(scores)) if len(scores) > 1 and total > 0 else 0.0
            rows.append({
                "kind": "split_or_label_change",
                "source_depth": d0,
                "target_depth": d1,
                "source_domain": source_domain,
                "target_domain": "",
                "top_targets": " ".join(str(row["target_domain"]) for row in top),
                "top_scores": " ".join(str(row["lineage_score"]) for row in top),
                "split_entropy": rounded(entropy),
                "candidate_status": "split_candidate" if entropy > 0.85 else "label_change_candidate" if top and top[0]["target_domain"] != source_domain else "single_lineage",
                "screen_only": True,
            })
        for target_domain in domains:
            incoming = [row for row in edge_rows if row["target_domain"] == target_domain and int(row["source_depth"]) == d0 and int(row["target_depth"]) == d1]
            top = sorted(incoming, key=lambda row: as_float(row.get("lineage_score"), -999.0), reverse=True)[:3]
            scores = [max(1e-8, as_float(row.get("lineage_score"), 0.0)) for row in top]
            total = sum(scores)
            entropy = -sum((s / total) * math.log(s / total) for s in scores) / math.log(len(scores)) if len(scores) > 1 and total > 0 else 0.0
            rows.append({
                "kind": "merge",
                "source_depth": d0,
                "target_depth": d1,
                "source_domain": "",
                "target_domain": target_domain,
                "top_sources": " ".join(str(row["source_domain"]) for row in top),
                "top_scores": " ".join(str(row["lineage_score"]) for row in top),
                "merge_entropy": rounded(entropy),
                "candidate_status": "merge_candidate" if entropy > 0.85 else "single_source",
                "screen_only": True,
            })
    path = ctx.path("tables", "split_merge_candidates.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Screen-only split, merge, and label-change candidates.")
    return rows


def run_with_residual_addition(bundle: bench.ModelBundle, prompt: str, depth: int, vector: Any, absolute_scale: float) -> Any:
    n_layers = bundle.anatomy.n_layers
    if not 0 <= depth <= n_layers:
        raise ValueError(f"stream depth must be in [0, {n_layers}], got {depth}")
    module = bundle.final_norm if depth == n_layers else bundle.blocks[depth]

    def add_hook(mod: Any, hook_args: tuple) -> Any:
        hidden = hook_args[0].clone()
        vec = unit_vector(vector.float()) * float(absolute_scale)
        hidden[0, -1] = hidden[0, -1] + vec.to(hidden.device, hidden.dtype)
        return (hidden,) + tuple(hook_args[1:])

    return bench._forward_logits(bundle, prompt, [(module, add_hook)])


def addition_noop_check(ctx: bench.RunContext, bundle: bench.ModelBundle, prompt: str, depth: int, vector: Any) -> dict[str, Any]:
    base = bench.run_with_residual_cache(bundle, prompt).final_logits_last
    edited = run_with_residual_addition(bundle, prompt, depth, vector, 0.0)
    max_abs = float((base - edited).abs().max())
    status = {
        "ok": max_abs <= RESIDUAL_ADD_NOOP_ATOL,
        "max_abs_logit_delta": max_abs,
        "atol": RESIDUAL_ADD_NOOP_ATOL,
        "depth": depth,
        "prompt": prompt,
    }
    path = ctx.path("diagnostics", "residual_addition_noop_check.json")
    bench.write_json(path, status)
    ctx.register_artifact(path, "diagnostic", "Lab-local zero-scale activation-addition no-op check.")
    if not status["ok"]:
        raise RuntimeError(f"Lab 30 residual-addition no-op failed: max |delta logit| {max_abs:.3g}")
    return status


def causal_transfer(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    rows: Sequence[CorpusRow],
    depths: Sequence[int],
    norm_by_depth: Mapping[int, float],
    node_state: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    domains = sorted({row.domain for row in rows})
    for domain in domains:
        domain_rows = [row for row in rows if row.domain == domain]
        marker = row_marker(domain_rows[0])
        contrast = row_contrast(domain_rows[0])
        marker_id = domain_rows[0].marker_id
        contrast_id = domain_rows[0].contrast_id
        neutral_prompt = str(domain_rows[0].labels.get("neutral_probe_prompt", "This passage is about"))
        base_logits = bench.run_with_residual_cache(bundle, neutral_prompt).final_logits_last
        base_margin = float(base_logits[marker_id] - base_logits[contrast_id])
        for depth in depths:
            direction = node_state[(domain, depth)]["direction"]
            rand = node_state[(domain, depth)]["random_direction"]
            abs_scale = float(norm_by_depth.get(depth, 1.0)) * TRANSFER_SCALE_FRACTION
            edited_logits = run_with_residual_addition(bundle, neutral_prompt, depth, direction, abs_scale)
            random_logits = run_with_residual_addition(bundle, neutral_prompt, depth, rand, abs_scale)
            edited_margin = float(edited_logits[marker_id] - edited_logits[contrast_id])
            random_margin = float(random_logits[marker_id] - random_logits[contrast_id])
            out.append({
                "domain": domain,
                "depth": depth,
                "claimable_depth": depth_claimable(bundle, depth),
                "neutral_probe_prompt": neutral_prompt,
                "marker_token": marker,
                "contrast_token": contrast,
                "marker_id": marker_id,
                "contrast_id": contrast_id,
                "scale_fraction_of_median_stream_norm": TRANSFER_SCALE_FRACTION,
                "absolute_scale": rounded(abs_scale),
                "base_marker_minus_contrast": rounded(base_margin),
                "edited_marker_minus_contrast": rounded(edited_margin),
                "random_marker_minus_contrast": rounded(random_margin),
                "transfer_gain": rounded(edited_margin - base_margin),
                "random_gain": rounded(random_margin - base_margin),
                "control_gap": rounded((edited_margin - base_margin) - (random_margin - base_margin)),
                "claim_scope": "marker_logit_only_not_semantic_steering",
            })
    path = ctx.path("tables", "causal_transfer_by_layer.csv")
    bench.write_csv_with_context(ctx, path, out)
    ctx.register_artifact(path, "table", "Marker-logit activation-addition transfer by domain and depth.")
    return out


def label_stability(
    ctx: bench.RunContext,
    rows: Sequence[CorpusRow],
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    domains = sorted({row.domain for row in rows})
    summary: list[dict[str, Any]] = []
    overlap: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for domain in domains:
        dnodes = [row for row in nodes if row["domain"] == domain and row.get("claimable_depth")]
        same_edges = [row for row in edges if row["source_domain"] == domain and row["target_domain"] == domain and row.get("claimable_edge")]
        outgoing = [row for row in edges if row["source_domain"] == domain and row.get("claimable_edge")]
        stable_edges = [row for row in same_edges if row.get("claim_candidate")]
        top_same_count = 0
        top_confusable_count = 0
        for edge in same_edges:
            competitors = [row for row in outgoing if int(row["source_depth"]) == int(edge["source_depth"]) and int(row["target_depth"]) == int(edge["target_depth"])]
            if competitors:
                best = max(competitors, key=lambda row: as_float(row.get("lineage_score"), -999.0))
                if best["target_domain"] == domain:
                    top_same_count += 1
                if best.get("is_confusable_edge"):
                    top_confusable_count += 1
        best_node = max(
            dnodes or [row for row in nodes if row["domain"] == domain],
            key=lambda row: (
                as_float(row.get("train_auc"), -999.0) - max(as_float(row.get("random_train_auc"), 0.5), 0.5),
                as_float(row.get("train_auc"), -999.0),
                -int(row.get("depth", 999)),
            ),
        )
        transfer = [row for row in transfer_rows if row["domain"] == domain and row.get("claimable_depth")]
        best_transfer_gap = safe_max([row.get("control_gap") for row in transfer])
        mean_eval_auc = safe_mean([row.get("eval_auc") for row in dnodes])
        mean_random_eval_auc = safe_mean([row.get("random_eval_auc") for row in dnodes])
        mean_same_score = safe_mean([row.get("lineage_score") for row in same_edges])
        mean_random_score = safe_mean([row.get("random_control_score") for row in same_edges])
        mean_confusable_score = safe_mean([row.get("confusable_control_score") for row in same_edges])
        mean_lineage_lift = mean_same_score - mean_random_score if math.isfinite(mean_same_score) and math.isfinite(mean_random_score) else float("nan")
        mean_confusable_gap = mean_same_score - mean_confusable_score if math.isfinite(mean_same_score) and math.isfinite(mean_confusable_score) else float("nan")
        label_survival_rate = top_same_count / len(same_edges) if same_edges else 0.0
        stable_edge_fraction = len(stable_edges) / len(same_edges) if same_edges else 0.0
        domain_ready = any(is_train(row) for row in rows if row.domain == domain) and any(is_eval(row) for row in rows if row.domain == domain)
        if not domain_ready:
            posture = "needs_more_split_data"
        elif mean_eval_auc < NODE_AUC_PASS:
            posture = "decodability_not_held_out"
        elif stable_edge_fraction >= 0.5 and label_survival_rate >= 0.5:
            posture = "recurring_lineage_supported"
        elif mean_confusable_gap < CONFUSABLE_PASS_GAP:
            posture = "confusable_limited_lineage"
        elif mean_lineage_lift < LINEAGE_PASS_LIFT:
            posture = "random_control_limited_lineage"
        else:
            posture = "lineage_needs_refinement"
        row = {
            "domain": domain,
            "confusable_domain": sorted({row_confusable(r) for r in rows if r.domain == domain})[0],
            "group_id": sorted({r.group_id for r in rows if r.domain == domain})[0],
            "best_depth": best_node["depth"],
            "selection_rule": "train_auc_lift_then_train_auc_then_earlier_claimable_depth",
            "best_train_auc": best_node.get("train_auc"),
            "best_random_train_auc": best_node.get("random_train_auc"),
            "best_eval_auc": best_node.get("eval_auc"),
            "best_random_eval_auc": best_node.get("random_eval_auc"),
            "mean_eval_auc": rounded(mean_eval_auc),
            "mean_random_eval_auc": rounded(mean_random_eval_auc),
            "mean_auc_lift_over_random": rounded(mean_eval_auc - mean_random_eval_auc if math.isfinite(mean_eval_auc) and math.isfinite(mean_random_eval_auc) else float("nan")),
            "mean_same_label_lineage_score": rounded(mean_same_score),
            "mean_random_control_score": rounded(mean_random_score),
            "mean_confusable_control_score": rounded(mean_confusable_score),
            "mean_lineage_lift_over_random": rounded(mean_lineage_lift),
            "mean_confusable_gap": rounded(mean_confusable_gap),
            "stable_edge_count": len(stable_edges),
            "claimable_same_edge_count": len(same_edges),
            "stable_edge_fraction": rounded(stable_edge_fraction),
            "label_survival_rate": rounded(label_survival_rate),
            "top_confusable_rate": rounded(top_confusable_count / len(same_edges) if same_edges else 0.0),
            "best_causal_transfer_gap": rounded(best_transfer_gap),
            "claim_posture": posture,
        }
        summary.append(row)
        overlap.append({
            "domain": domain,
            "model_a": "loaded_model",
            "model_b": "same_model_cross_layer",
            "control_model": "deterministic_random_direction",
            "same_model_overlap_score": rounded(mean_same_score),
            "random_control_overlap_score": rounded(mean_random_score),
            "confusable_control_overlap_score": rounded(mean_confusable_score),
            "overlap_lift_over_random": rounded(mean_lineage_lift),
            "overlap_lift_over_confusable": rounded(mean_confusable_gap),
            "external_cross_model_status": "not_run_in_default_lab30",
            "claim_allowed": "same_model_cross_layer_schema_only",
        })
        evidence.append({
            **row,
            "evidence_tag": "DECODE+ATTR" if posture == "recurring_lineage_supported" else "DECODE+ATTR,AUDIT",
            "allowed_claim": (
                "recurring supervised prototype-direction handle" if posture == "recurring_lineage_supported"
                else "negative or refinement result; do not use lineage identity language"
            ),
            "forbidden_claim": "same concept everywhere in the model",
        })
    summary_path = ctx.path("tables", "label_stability_summary.csv")
    bench.write_csv_with_context(ctx, summary_path, summary)
    ctx.register_artifact(summary_path, "table", "Domain-level label stability and lineage verdicts.")
    overlap_path = ctx.path("tables", "cross_model_feature_overlap.csv")
    bench.write_csv_with_context(ctx, overlap_path, overlap)
    ctx.register_artifact(overlap_path, "table", "Same-model cross-layer overlap using the future cross-model schema; external comparison not run.")
    evidence_path = ctx.path("tables", "feature_lineage_evidence_matrix.csv")
    bench.write_csv_with_context(ctx, evidence_path, evidence)
    ctx.register_artifact(evidence_path, "table", "Claim-ready evidence matrix for Lab 30 feature lineage.")
    standard_evidence_path = ctx.path("tables", "evidence_matrix.csv")
    bench.write_csv_with_context(ctx, standard_evidence_path, evidence)
    ctx.register_artifact(standard_evidence_path, "table", "Standard evidence-matrix alias for Lab 30.")
    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, evidence)
    ctx.register_artifact(results_path, "table", "Alias of the Lab 30 evidence matrix for the standard results slot.")
    confusable_rows = [
        {
            "domain": row["domain"],
            "confusable_domain": row["confusable_domain"],
            "mean_same_label_lineage_score": row["mean_same_label_lineage_score"],
            "mean_confusable_control_score": row["mean_confusable_control_score"],
            "mean_random_control_score": row["mean_random_control_score"],
            "mean_confusable_gap": row["mean_confusable_gap"],
            "mean_lineage_lift_over_random": row["mean_lineage_lift_over_random"],
            "claim_posture": row["claim_posture"],
        }
        for row in summary
    ]
    conf_path = ctx.path("tables", "confusable_control_ladder.csv")
    bench.write_csv_with_context(ctx, conf_path, confusable_rows)
    ctx.register_artifact(conf_path, "table", "Domain-level same-label lineage versus confusable and random controls.")
    metrics = {
        "n_domains": len(domains),
        "supported_domains": sum(1 for row in summary if row["claim_posture"] == "recurring_lineage_supported"),
        "mean_eval_auc": rounded(safe_mean([row["mean_eval_auc"] for row in summary])),
        "mean_lineage_lift_over_random": rounded(safe_mean([row["mean_lineage_lift_over_random"] for row in summary])),
        "mean_confusable_gap": rounded(safe_mean([row["mean_confusable_gap"] for row in summary])),
        "verdicts": {row["domain"]: row["claim_posture"] for row in summary},
    }
    return summary, overlap, evidence, metrics


def build_counterexamples(
    ctx: bench.RunContext,
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in nodes:
        if not node.get("claimable_depth"):
            continue
        eval_auc = as_float(node.get("eval_auc"))
        random_auc = as_float(node.get("random_eval_auc"))
        if math.isfinite(eval_auc) and math.isfinite(random_auc) and eval_auc <= random_auc + NODE_RANDOM_LIFT_PASS:
            out.append({
                "kind": "node_random_control_match",
                "severity": rounded((random_auc + NODE_RANDOM_LIFT_PASS) - eval_auc),
                "domain": node["domain"],
                "depth": node["depth"],
                "node_id": node["node_id"],
                "eval_auc": node["eval_auc"],
                "control_value": node["random_eval_auc"],
                "artifact_row": "tables/feature_lineage_nodes.csv",
                "interpretation": "Direction is not held-out label-valid above random control at this depth.",
            })
    for edge in edges:
        if not edge.get("same_label") or not edge.get("claimable_edge"):
            continue
        score = as_float(edge.get("lineage_score"))
        rand = as_float(edge.get("random_control_score"))
        conf = as_float(edge.get("confusable_control_score"))
        if math.isfinite(score) and math.isfinite(rand) and score - rand < LINEAGE_PASS_LIFT:
            out.append({
                "kind": "edge_random_control_too_close",
                "severity": rounded(LINEAGE_PASS_LIFT - (score - rand)),
                "domain": edge["source_domain"],
                "edge_id": edge["edge_id"],
                "lineage_score": edge["lineage_score"],
                "control_value": edge["random_control_score"],
                "artifact_row": "tables/feature_lineage_edges.csv",
                "interpretation": "Same-label edge does not beat random-direction control enough.",
            })
        if math.isfinite(score) and math.isfinite(conf) and score - conf < CONFUSABLE_PASS_GAP:
            out.append({
                "kind": "edge_confusable_control_too_close",
                "severity": rounded(CONFUSABLE_PASS_GAP - (score - conf)),
                "domain": edge["source_domain"],
                "edge_id": edge["edge_id"],
                "lineage_score": edge["lineage_score"],
                "control_value": edge["confusable_control_score"],
                "confusable_domain": edge["confusable_target_domain"],
                "artifact_row": "tables/feature_lineage_edges.csv",
                "interpretation": "Confusable-domain edge is close enough to shrink the lineage claim.",
            })
    for row in transfer_rows:
        gap = as_float(row.get("control_gap"))
        if row.get("claimable_depth") and math.isfinite(gap) and gap < 0:
            out.append({
                "kind": "marker_transfer_random_beats_direction",
                "severity": rounded(abs(gap)),
                "domain": row["domain"],
                "depth": row["depth"],
                "lineage_score": "",
                "control_value": row.get("random_gain"),
                "artifact_row": "tables/causal_transfer_by_layer.csv",
                "interpretation": "Random direction moved the marker-token margin more than the prototype direction.",
            })
    out.sort(key=lambda row: as_float(row.get("severity"), 0.0), reverse=True)
    path = ctx.path("tables", "feature_lineage_counterexamples.csv")
    bench.write_csv_with_context(ctx, path, out)
    ctx.register_artifact(path, "table", "Automatic counterexamples limiting feature-lineage claims.")
    alias_path = ctx.path("tables", "counterexamples.csv")
    bench.write_csv_with_context(ctx, alias_path, out)
    ctx.register_artifact(alias_path, "table", "Standard counterexample alias for Lab 30.")
    return out


def save_state(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    rows: Sequence[CorpusRow],
    depths: Sequence[int],
    node_state: Mapping[tuple[str, int], Mapping[str, Any]],
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> None:
    import torch

    dictionary = {
        "lab": LAB_ID,
        "model": bundle.anatomy.model_id,
        "feature_kind": "supervised_prototype_direction",
        "depths": list(depths),
        "domains": sorted({row.domain for row in rows}),
        "directions": {f"{domain}@d{depth}": state["direction"].float().cpu() for (domain, depth), state in node_state.items()},
        "random_controls": {f"{domain}@d{depth}": state["random_direction"].float().cpu() for (domain, depth), state in node_state.items()},
        "non_claim": "This is a lightweight prototype dictionary, not an SAE/crosscoder dictionary.",
    }
    path = ctx.path("state", "cross_layer_dictionary.pt")
    torch.save(dictionary, path)
    ctx.register_artifact(path, "state", "Prototype cross-layer dictionary of supervised directions and random controls.")
    meta = {
        "lab": LAB_ID,
        "model": bundle.anatomy.model_id,
        "n_layers": bundle.anatomy.n_layers,
        "d_model": bundle.anatomy.d_model,
        "depths": list(depths),
        "domains": sorted({row.domain for row in rows}),
        "feature_kind": "supervised_prototype_direction",
        "fitting_split": "train rows only when available",
        "position": "final prompt token",
        "state_file": "state/cross_layer_dictionary.pt",
        "non_claims": [
            "not an SAE feature dictionary",
            "not a trained cross-layer dictionary",
            "not evidence of feature identity",
            "not an external cross-model comparison",
        ],
    }
    meta_path = ctx.path("state", "cross_layer_dictionary_metadata.json")
    bench.write_json(meta_path, meta)
    ctx.register_artifact(meta_path, "state", "Human-readable metadata for the Lab 30 prototype dictionary.")
    graph = {
        "nodes": list(nodes),
        "edges": [dict(row) for row in edges if row.get("same_label") or row.get("claim_candidate") or row.get("is_confusable_edge")],
        "non_claim": "Graph edges are candidate recurring directions, not proof of feature identity.",
    }
    graph_path = ctx.path("state", "lineage_graph.json")
    bench.write_json(graph_path, graph)
    ctx.register_artifact(graph_path, "state", "Lineage graph JSON for same-label and confusable edges.")
    marker_map = {
        domain: {
            "marker_token": sorted({row_marker(row) for row in rows if row.domain == domain})[0],
            "contrast_token": sorted({row_contrast(row) for row in rows if row.domain == domain})[0],
            "confusable_domain": sorted({row_confusable(row) for row in rows if row.domain == domain})[0],
        }
        for domain in sorted({row.domain for row in rows})
    }
    marker_path = ctx.path("state", "domain_markers.json")
    bench.write_json(marker_path, marker_map)
    ctx.register_artifact(marker_path, "state", "Domain marker, contrast, and confusable labels for the marker-transfer probe.")


# ---------------------------------------------------------------------------
# Cards, summaries, and claims
# ---------------------------------------------------------------------------


def write_self_check_status(
    ctx: bench.RunContext,
    data_info: Mapping[str, Any],
    token_rows: Sequence[Mapping[str, Any]],
    corpus_rows: Sequence[Mapping[str, Any]],
    noop_status: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    status = {
        "tokenization_kept": sum(1 for row in token_rows if row.get("kept") is True),
        "tokenization_dropped": sum(1 for row in token_rows if row.get("kept") is False),
        "science_ready_data": bool(data_info.get("science_ready_data")),
        "domains_missing_eval": [row["domain"] for row in corpus_rows if not row.get("science_ready_domain")],
        "residual_addition_noop_ok": bool(noop_status.get("ok")),
        "n_claimable_nodes": sum(1 for row in nodes if row.get("claimable_node")),
        "n_claim_candidate_edges": sum(1 for row in edges if row.get("claim_candidate")),
        "ok": True,
    }
    status["ok"] = bool(status["tokenization_kept"] > 0 and status["residual_addition_noop_ok"] and not status["domains_missing_eval"])
    path = ctx.path("diagnostics", "self_check_status.json")
    bench.write_json(path, status)
    ctx.register_artifact(path, "diagnostic", "Lab 30 local self-check status.")
    return status




def write_safety_status(ctx: bench.RunContext) -> dict[str, Any]:
    status = {
        "safe_scope": "benign public and synthetic domain sentences",
        "generation": "not_used",
        "private_data": "not_used",
        "persistent_weight_edit": "not_run",
        "external_cross_model_comparison": "not_run_in_default_lab30",
        "activation_addition_scope": "marker-logit probe only on a neutral prompt",
        "safety_status": "low_risk_forward_pass_lab",
    }
    path = ctx.path("diagnostics", "safety_status.json")
    bench.write_json(path, status)
    ctx.register_artifact(path, "diagnostic", "Safety and scope status for Lab 30.")
    return status

def write_method_card(ctx: bench.RunContext, bundle: bench.ModelBundle, data_info: Mapping[str, Any], summary: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Lab 30 method card",
        "",
        "This lab uses supervised prototype directions. It does not train an SAE, transcoder, sparse crosscoder, or external cross-model mapper.",
        "",
        f"- model: `{bundle.anatomy.model_id}`",
        f"- data source: `{data_info.get('data_source')}`",
        f"- science-ready frozen data: `{bool(data_info.get('science_ready_data'))}`",
        "- feature unit: train-split domain mean-minus-rest direction at a residual depth",
        "- node evidence: held-out domain AUC above deterministic random direction",
        "- edge evidence: adjacent-depth cosine, eval activation correlation, eval top-context overlap, and endpoint AUC",
        "- controls: deterministic random directions and confusable-domain competitors",
        "- cross-model status: default run exports same-model overlap versus random controls; external cross-model comparison is not run",
        "- evidence rung: `DECODE + ATTR`, with a narrow marker-logit activation-addition probe",
        "- forbidden claim: this is the same concept everywhere in the model",
        "",
        "| domain | best depth | eval AUC | stable edges | survival | confusable gap | posture |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['domain']}` | {row['best_depth']} | {row['mean_eval_auc']} | "
            f"{row['stable_edge_count']}/{row['claimable_same_edge_count']} | {row['label_survival_rate']} | "
            f"{row['mean_confusable_gap']} | `{row['claim_posture']}` |"
        )
    path = ctx.path("method_card.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 30 method contract and lineage verdicts.")


def write_operationalization_audit(ctx: bench.RunContext, summary: Sequence[Mapping[str, Any]], counterexamples: Sequence[Mapping[str, Any]]) -> None:
    supported = sum(1 for row in summary if row["claim_posture"] == "recurring_lineage_supported")
    result = "passed" if supported == len(summary) and not counterexamples else "mixed" if supported else "failed"
    lines = [
        "# Lab 30 operationalization audit",
        "",
        "```yaml",
        "headline_claim: \"a domain feature persists across layers\"",
        "cheap_explanation: \"a supervised direction tracks surface vocabulary, global residual alignment, or a confusable domain rather than a stable feature\"",
        "killer_control: \"held-out AUC, random-direction edges, confusable-domain edges, top-context overlap, and counterexample logging\"",
        f"result: \"{result}\"",
        "claim_allowed: \"handle\"",
        "```",
        "",
        "## What the measurement can say",
        "",
        "A supervised prototype direction recurred across adjacent residual depths with stronger label validity, activation-score correlation, direction similarity, and top-context overlap than random and confusable controls.",
        "",
        "## What it cannot say",
        "",
        "It cannot identify an SAE feature, prove monosemanticity, show semantic identity across layers, or report an external cross-model result.",
        "",
        "## Cheap explanations and controls",
        "",
        "| Cheap explanation | Control | Evidence that would make the cheap explanation win |",
        "|---|---|---|",
        "| surface-token direction | confusable-domain edge | confusable edge matches same-label edge |",
        "| global residual alignment | random-direction edge | random edge has similar lineage score |",
        "| decodable but not recurring | held-out node AUC plus edge gate | node AUC high but edge gates fail |",
        "| recurring but not label-valid | held-out AUC gate | same-label geometry high but eval AUC low |",
        "| marker-token steering | separate transfer table | transfer works while lineage evidence fails |",
        "| cross-model overclaim | explicit same-model placeholder | external_cross_model_status is not_run |",
        "",
        "## Verdicts",
        "",
    ]
    for row in summary:
        lines.append(
            f"- `{row['domain']}`: `{row['claim_posture']}`; eval AUC {row['mean_eval_auc']}, "
            f"random lift {row['mean_lineage_lift_over_random']}, confusable gap {row['mean_confusable_gap']}."
        )
    lines += ["", "## Counterexamples", ""]
    if counterexamples:
        for row in counterexamples[:12]:
            lines.append(f"- `{row['kind']}` for `{row.get('domain', '')}`: {row.get('interpretation', '')}")
    else:
        lines.append("- No automatic counterexamples crossed the configured thresholds. Replicate before broadening the claim.")
    lines += [
        "",
        "## Allowed language",
        "",
        "- `This supervised prototype direction is a recurring lineage handle under the Lab 30 corpus and controls.`",
        "- `The same-label edge beat the random and confusable controls at these depths.`",
        "",
        "## Forbidden language",
        "",
        "- `This is the same concept everywhere in the model.`",
        "- `This prototype direction is an SAE feature.`",
        "- `This default run proves cross-model overlap.`",
        "- `The marker-logit probe proves semantic steering.`",
    ]
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Controls, cheap explanations, counterexamples, and claim grammar for Lab 30.")


def write_run_summary(ctx: bench.RunContext, data_info: Mapping[str, Any], metrics: Mapping[str, Any], summary: Sequence[Mapping[str, Any]], counterexamples: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Lab 30 run summary: feature lineage without feature identity",
        "",
        "## Run identity",
        "",
        f"- data rows: {data_info['n_rows_selected']} selected from `{pathlib.Path(str(data_info['data_path'])).name}`",
        f"- data source: `{data_info['data_source']}`",
        f"- science scope: {data_info['science_scope']}",
        f"- domains: `{data_info['domains']}`",
        f"- supported domains: `{metrics['supported_domains']}` / `{metrics['n_domains']}`",
        f"- automatic counterexamples: {len(counterexamples)}",
        "",
        "## What behavior was measured?",
        "",
        "The lab measured domain-label decodability from residual-stream prototype directions, adjacent-depth lineage scores, confusable controls, and a narrow marker-token activation-addition margin.",
        "",
        "## Headline verdicts",
        "",
        "| domain | best depth | eval AUC | stable edges | confusable gap | posture |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['domain']}` | {row['best_depth']} | {row['mean_eval_auc']} | "
            f"{row['stable_edge_count']}/{row['claimable_same_edge_count']} | {row['mean_confusable_gap']} | `{row['claim_posture']}` |"
        )
    lines += [
        "",
        "## Reading order",
        "",
        "1. `method_card.md` for the method contract and verdicts.",
        "2. `diagnostics/tokenization_gate.csv` and `tables/corpus_manifest.csv` for data validity.",
        "3. `tables/feature_lineage_nodes.csv` for held-out node AUC.",
        "4. `tables/feature_lineage_edges.csv` for random and confusable edge controls.",
        "5. `tables/feature_lineage_evidence_matrix.csv` for claim posture.",
        "6. `tables/feature_lineage_counterexamples.csv` before writing a positive claim.",
        "7. `tables/cross_model_feature_overlap.csv` to confirm the default run is not an external cross-model result.",
        "8. `operationalization_audit.md` for allowed and forbidden language.",
        "",
        "## Smallest surviving claim",
        "",
        "A supported row means a supervised domain direction recurred across adjacent residual depths above random and confusable controls. It does not mean the same monosemantic feature exists everywhere.",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 30 run summary and artifact reading path.")


def write_plot_guide(ctx: bench.RunContext) -> None:
    rows = [
        {"plot": "plots/feature_lineage_dashboard.png", "read_for": "Domain verdicts, eval AUC, stable edges, confusable gap, marker transfer.", "non_claim": "Not feature identity."},
        {"plot": "plots/node_auc_by_depth.png", "read_for": "Whether node directions are held-out label-valid by depth.", "non_claim": "AUC alone is not lineage."},
        {"plot": "plots/cross_layer_feature_graph.png", "read_for": "Same-label adjacent-depth lineage score traces.", "non_claim": "Graph edges are candidate handles."},
        {"plot": "plots/lineage_similarity_matrix.png", "read_for": "Cross-domain lineage scores and confusable leakage.", "non_claim": "High off-diagonal scores may be a confound."},
        {"plot": "plots/confusable_control_ladder.png", "read_for": "Whether same-label edges beat confusable and random controls.", "non_claim": "No broad claim if the confusable control is close."},
        {"plot": "plots/feature_split_merge_atlas.png", "read_for": "Split/merge entropy screens.", "non_claim": "Entropy is a hypothesis screen, not proof."},
        {"plot": "plots/label_stability_ladder.png", "read_for": "Label survival by domain.", "non_claim": "Label survival is not semantic identity."},
        {"plot": "plots/cross_model_feature_overlap.png", "read_for": "Same-model overlap versus random controls using cross-model schema.", "non_claim": "External cross-model comparison is not run."},
        {"plot": "plots/causal_transfer_by_layer.png", "read_for": "Marker-logit activation-addition transfer by layer.", "non_claim": "Marker logits are a narrow causal probe."},
    ]
    path = ctx.path("tables", "plot_reading_guide.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Plot reading guide for Lab 30.")
    plot_path = ctx.path("plots", "plot_reading_guide.csv")
    bench.write_csv_with_context(ctx, plot_path, rows)
    ctx.register_artifact(plot_path, "table", "Copy of the Lab 30 plot reading guide next to the plot files.")


def write_claims(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]]) -> None:
    claims = []
    run_name = ctx.run_dir.name
    for i, row in enumerate(evidence, start=1):
        if row["claim_posture"] == "recurring_lineage_supported":
            text = (
                f"For domain `{row['domain']}`, Lab 30 found a recurring supervised prototype-direction lineage: "
                f"mean held-out node AUC {row['mean_eval_auc']}, stable edges {row['stable_edge_count']}/{row['claimable_same_edge_count']}, "
                f"random lift {row['mean_lineage_lift_over_random']}, and confusable gap {row['mean_confusable_gap']}. "
                "This is a lineage-handle claim, not feature identity."
            )
        else:
            text = (
                f"For domain `{row['domain']}`, Lab 30 did not earn strong lineage language: posture `{row['claim_posture']}`, "
                f"mean held-out AUC {row['mean_eval_auc']}, random lift {row['mean_lineage_lift_over_random']}, "
                f"confusable gap {row['mean_confusable_gap']}."
            )
        claims.append({
            "id": f"{LAB_ID}-C{i}",
            "tag": "DECODE+ATTR,AUDIT",
            "text": text,
            "artifact": f"runs/{run_name}/tables/feature_lineage_evidence_matrix.csv",
            "falsifier": "A balanced held-out corpus where eval AUC collapses, or confusable/random edges match the same-label lineage scores.",
        })
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def write_plots(
    ctx: bench.RunContext,
    summary: Sequence[Mapping[str, Any]],
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
    overlap: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
) -> None:
    write_plot_guide(ctx)
    if ctx.args.no_plots:
        return
    import matplotlib.pyplot as plt
    import numpy as np

    domains = [str(row["domain"]) for row in summary]
    x = np.arange(len(domains))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Lab 30 feature lineage dashboard", fontsize=14, fontweight="bold")
    axes[0, 0].bar(x, [as_float(row.get("mean_eval_auc"), 0.0) for row in summary])
    axes[0, 0].axhline(NODE_AUC_PASS, linestyle="--", linewidth=0.8)
    axes[0, 0].set_xticks(x, domains, rotation=35, ha="right")
    axes[0, 0].set_title("Held-out node AUC")

    axes[0, 1].bar(x, [as_float(row.get("stable_edge_fraction"), 0.0) for row in summary])
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_xticks(x, domains, rotation=35, ha="right")
    axes[0, 1].set_title("Stable edge fraction")

    axes[0, 2].bar(x, [as_float(row.get("label_survival_rate"), 0.0) for row in summary])
    axes[0, 2].set_ylim(0, 1.05)
    axes[0, 2].set_xticks(x, domains, rotation=35, ha="right")
    axes[0, 2].set_title("Label survival")

    axes[1, 0].bar(x, [as_float(row.get("mean_lineage_lift_over_random"), 0.0) for row in summary])
    axes[1, 0].axhline(LINEAGE_PASS_LIFT, linestyle="--", linewidth=0.8)
    axes[1, 0].set_xticks(x, domains, rotation=35, ha="right")
    axes[1, 0].set_title("Lift over random")

    axes[1, 1].bar(x, [as_float(row.get("mean_confusable_gap"), 0.0) for row in summary])
    axes[1, 1].axhline(CONFUSABLE_PASS_GAP, linestyle="--", linewidth=0.8)
    axes[1, 1].set_xticks(x, domains, rotation=35, ha="right")
    axes[1, 1].set_title("Gap over confusable")

    axes[1, 2].bar(x, [as_float(row.get("best_causal_transfer_gap"), 0.0) for row in summary])
    axes[1, 2].axhline(0, linewidth=0.8)
    axes[1, 2].set_xticks(x, domains, rotation=35, ha="right")
    axes[1, 2].set_title("Best marker-transfer gap")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    bench.save_figure(ctx, fig, "feature_lineage_dashboard.png", "Lab 30 feature-lineage evidence dashboard.")

    fig, ax = plt.subplots(figsize=(9, 5))
    depths = sorted({int(row["depth"]) for row in nodes})
    for domain in domains:
        dnodes = sorted([row for row in nodes if row["domain"] == domain], key=lambda row: int(row["depth"]))
        ax.plot([int(row["depth"]) for row in dnodes], [as_float(row.get("eval_auc"), float("nan")) for row in dnodes], marker="o", label=domain)
        ax.plot([int(row["depth"]) for row in dnodes], [as_float(row.get("random_eval_auc"), float("nan")) for row in dnodes], linestyle=":", linewidth=0.8)
    ax.axhline(NODE_AUC_PASS, linestyle="--", linewidth=0.8)
    ax.set_xticks(depths)
    ax.set_xlabel("stream depth")
    ax.set_ylabel("eval AUC")
    ax.set_title("Node AUC by depth; dotted lines are random controls")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "node_auc_by_depth.png", "Held-out node AUC by depth with random controls.")

    same_edges = [row for row in edges if row.get("same_label")]
    fig, ax = plt.subplots(figsize=(9, 5))
    for domain in domains:
        d_edges = sorted([row for row in same_edges if row["source_domain"] == domain], key=lambda row: int(row["source_depth"]))
        ax.plot([int(row["source_depth"]) for row in d_edges], [as_float(row.get("lineage_score"), float("nan")) for row in d_edges], marker="o", label=domain)
    ax.axhline(LINEAGE_PASS_SCORE, linestyle="--", linewidth=0.8)
    ax.set_xlabel("source depth")
    ax.set_ylabel("same-label lineage score")
    ax.set_title("Cross-layer feature graph")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "cross_layer_feature_graph.png", "Same-label adjacent-depth lineage scores.")

    mat = np.zeros((len(domains), len(domains)))
    for i, sd in enumerate(domains):
        for j, td in enumerate(domains):
            vals = [row.get("lineage_score") for row in edges if row["source_domain"] == sd and row["target_domain"] == td and row.get("claimable_edge")]
            mat[i, j] = safe_mean(vals, default=0.0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(domains)), domains, rotation=35, ha="right")
    ax.set_yticks(range(len(domains)), domains)
    ax.set_title("Mean claimable lineage score matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "lineage_similarity_matrix.png", "Cross-domain lineage similarity matrix.")

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    ax.bar(x - width, [as_float(row.get("mean_same_label_lineage_score"), 0.0) for row in summary], width, label="same label")
    ax.bar(x, [as_float(row.get("mean_confusable_control_score"), 0.0) for row in summary], width, label="confusable")
    ax.bar(x + width, [as_float(row.get("mean_random_control_score"), 0.0) for row in summary], width, label="random")
    ax.axhline(LINEAGE_PASS_SCORE, linestyle="--", linewidth=0.8)
    ax.set_xticks(x, domains, rotation=35, ha="right")
    ax.set_ylabel("mean claimable edge score")
    ax.set_title("Confusable control ladder")
    ax.legend(frameon=False)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "confusable_control_ladder.png", "Same-label lineage versus confusable and random controls.")

    split_vals = [as_float(row.get("split_entropy"), 0.0) for row in split_rows if row["kind"] == "split_or_label_change"]
    merge_vals = [as_float(row.get("merge_entropy"), 0.0) for row in split_rows if row["kind"] == "merge"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(split_vals, bins=8, alpha=0.7, label="split entropy")
    ax.hist(merge_vals, bins=8, alpha=0.7, label="merge entropy")
    ax.set_title("Feature split/merge atlas")
    ax.set_xlabel("entropy")
    ax.legend(frameon=False)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "feature_split_merge_atlas.png", "Screen-only split/merge entropy atlas.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(domains, [as_float(row.get("label_survival_rate"), 0.0) for row in summary])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("label survival rate")
    ax.set_title("Label stability ladder")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "label_stability_ladder.png", "Label stability ladder.")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, [as_float(row.get("same_model_overlap_score"), 0.0) for row in overlap], 0.4, label="same-model")
    ax.bar(x + 0.2, [as_float(row.get("random_control_overlap_score"), 0.0) for row in overlap], 0.4, label="random")
    ax.set_xticks(x, domains, rotation=35, ha="right")
    ax.set_title("Cross-model feature overlap placeholder")
    ax.legend(frameon=False)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "cross_model_feature_overlap.png", "Same-model overlap versus random controls; external cross-model not run.")

    fig, ax = plt.subplots(figsize=(9, 5))
    transfer_depths = sorted({int(row["depth"]) for row in transfer_rows})
    for domain in domains:
        d_rows = sorted([row for row in transfer_rows if row["domain"] == domain], key=lambda row: int(row["depth"]))
        ax.plot([int(row["depth"]) for row in d_rows], [as_float(row.get("control_gap"), float("nan")) for row in d_rows], marker="o", label=domain)
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(transfer_depths)
    ax.set_xlabel("stream depth")
    ax.set_ylabel("marker-transfer control gap")
    ax.set_title("Causal transfer by layer")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "causal_transfer_by_layer.png", "Marker-logit activation-addition transfer by layer.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    rows, data_info = load_rows(ctx)
    print(f"[lab30] loaded {data_info['n_rows_selected']}/{data_info['n_rows_file']} rows from {pathlib.Path(str(data_info['data_path'])).name}")
    safety_status = write_safety_status(ctx)
    rows, token_rows = tokenization_gate(ctx, bundle, rows)
    corpus_rows = corpus_manifest(ctx, rows)
    data_info = {**data_info, "n_rows_after_tokenization": len(rows), "tokenization_kept": len(rows), "tokenization_dropped": len(token_rows) - len(rows)}
    manifest_path = ctx.path("diagnostics", "data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Lab 30 data manifest and first-pass method scope.")

    bench.run_hook_parity_check(ctx, bundle, rows[0].text)
    first = bench.run_with_residual_cache(bundle, rows[0].text)
    bench.run_lens_self_check(ctx, bundle, first)
    bench.run_patch_noop_check(ctx, bundle, rows[0].text)

    captures, depths, norm_by_depth = capture_corpus(ctx, bundle, rows)
    first_claim_depth = next((d for d in depths if depth_claimable(bundle, d)), depths[0])
    noop_status = addition_noop_check(ctx, bundle, rows[0].text, first_claim_depth, captures[rows[0].row_id].streams[first_claim_depth, -1])

    nodes, node_state = build_nodes(ctx, bundle, rows, captures, depths)
    edges = build_edges(ctx, bundle, rows, depths, node_state)
    domains = sorted({row.domain for row in rows})
    split_rows = split_merge_tables(ctx, edges, depths, domains)
    transfer_rows = causal_transfer(ctx, bundle, rows, depths, norm_by_depth, node_state)
    summary, overlap, evidence, metrics = label_stability(ctx, rows, nodes, edges, transfer_rows)
    counterexamples = build_counterexamples(ctx, nodes, edges, transfer_rows)
    self_check = write_self_check_status(ctx, data_info, token_rows, corpus_rows, noop_status, nodes, edges)
    save_state(ctx, bundle, rows, depths, node_state, nodes, edges)

    jsonl_path = ctx.path("results.jsonl")
    write_jsonl(jsonl_path, [{**ctx.table_context(), **row} for row in evidence])
    ctx.register_artifact(jsonl_path, "table", "JSONL copy of the Lab 30 evidence matrix.")

    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, {**metrics, "data": data_info, "depths": list(depths), "self_check_status": self_check, "safety_status": safety_status, "n_nodes": len(nodes), "n_edges": len(edges), "n_counterexamples": len(counterexamples)})
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 30 metrics and verdicts.")

    write_method_card(ctx, bundle, data_info, summary)
    write_operationalization_audit(ctx, summary, counterexamples)
    write_run_summary(ctx, data_info, metrics, summary, counterexamples)
    write_claims(ctx, evidence)
    write_plots(ctx, summary, nodes, edges, split_rows, overlap, transfer_rows)
    print(f"[lab30] wrote {len(nodes)} nodes, {len(edges)} edges, {len(summary)} domain verdicts, and {len(counterexamples)} counterexamples")
