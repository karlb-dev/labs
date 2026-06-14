"""Lab 30: Cross-layer and cross-model feature geometry.

This first implementation studies feature lineage without training an SAE or
crosscoder. It builds supervised domain prototype directions at multiple
residual depths, then audits whether same-label directions persist across
layers better than random and confusable-domain controls. The exported state is
a lightweight "prototype dictionary", not a sparse autoencoder dictionary.

Evidence level: DECODE + ATTR, with a small activation-addition transfer probe.
Claims are scoped to recurring supervised directions on the frozen corpus.
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
from typing import Any, Mapping, Sequence

import interp_bench as bench

LAB_ID = "L30"
DATA_FILE = "feature_lineage_corpus.csv"
PROMPT_SET_CAPS = {"small": 32, "medium": 32, "full": 0}
LINEAGE_PASS_GAP = 0.08
LINEAGE_PASS_SCORE = 0.62
CAUSAL_SCALE = 0.65
NEUTRAL_PROMPT = "This passage is about"


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
    pairs = [(int(y), float(s)) for y, s in zip(labels, scores) if math.isfinite(float(s))]
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
    import torch.nn.functional as F

    return float(F.cosine_similarity(a.float(), b.float(), dim=0).detach().cpu())


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(a)
    sb = set(b)
    return len(sa & sb) / len(sa | sb) if sa or sb else 0.0


def data_path(args: Any) -> pathlib.Path:
    prompt_set = str(getattr(args, "prompt_set", "") or "")
    candidate = pathlib.Path(prompt_set)
    if prompt_set not in PROMPT_SET_CAPS and candidate.suffix.lower() in {".csv", ".tsv"}:
        return candidate if candidate.is_absolute() else (bench.COURSE_ROOT / candidate).resolve()
    return bench.COURSE_ROOT / "data" / DATA_FILE


def load_rows(ctx: bench.RunContext) -> tuple[list[CorpusRow], dict[str, Any]]:
    path = data_path(ctx.args)
    raw_rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    rows = [
        CorpusRow(
            row_id=row["row_id"],
            family=row["family"],
            domain=row["domain"],
            source_lab=row["source_lab"],
            text=row["text"],
            group_id=row["group_id"],
            split=row["split"],
            labels=json.loads(row["labels_json"]),
        )
        for row in raw_rows
    ]
    cap = PROMPT_SET_CAPS.get(str(ctx.args.prompt_set), 0)
    if cap:
        rows = rows[:cap]
    if int(ctx.args.max_examples or 0) > 0:
        rows = rows[: int(ctx.args.max_examples)]
    info = {
        "data_path": str(path),
        "sha256": file_sha256(path),
        "n_rows_file": len(raw_rows),
        "n_rows_selected": len(rows),
        "domains": dict(Counter(r.domain for r in rows)),
        "source_labs": dict(Counter(r.source_lab for r in rows)),
        "splits": dict(Counter(r.split for r in rows)),
        "science_ready": True,
        "science_scope": "supervised prototype directions for cross-layer lineage; no trained SAE/crosscoder in first pass",
    }
    return rows, info


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def tokenization_gate(ctx: bench.RunContext, bundle: bench.ModelBundle, rows: list[CorpusRow]) -> list[CorpusRow]:
    tok = bundle.tokenizer
    kept: list[CorpusRow] = []
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        problems: list[str] = []
        ids = token_ids(tok, row.text)
        marker_ids = token_ids(tok, str(row.labels.get("marker_token", "")))
        contrast_ids = token_ids(tok, str(row.labels.get("contrast_token", "")))
        if not ids:
            problems.append("empty_text")
        if len(marker_ids) != 1:
            problems.append(f"marker_token_count={len(marker_ids)}")
        if len(contrast_ids) != 1:
            problems.append(f"contrast_token_count={len(contrast_ids)}")
        if not problems:
            kept.append(row)
        out_rows.append({
            "row_id": row.row_id,
            "domain": row.domain,
            "split": row.split,
            "n_tokens": len(ids),
            "marker_token": row.labels.get("marker_token", ""),
            "marker_token_count": len(marker_ids),
            "contrast_token": row.labels.get("contrast_token", ""),
            "contrast_token_count": len(contrast_ids),
            "kept": not problems,
            "problems": ";".join(problems),
        })
    path = ctx.path("diagnostics", "tokenization_gate.csv")
    bench.write_csv_with_context(ctx, path, out_rows)
    ctx.register_artifact(path, "diagnostic", "Text and marker-token audit for Lab 30 corpus.")
    if not kept:
        raise RuntimeError("Lab 30 tokenization gate dropped every row.")
    return kept


def coarse_depths(n_layers: int, prompt_set: str) -> list[int]:
    if prompt_set == "full":
        return list(range(n_layers + 1))
    return sorted({0, max(1, n_layers // 4), max(1, n_layers // 2), max(1, (3 * n_layers) // 4), n_layers})


def capture_corpus(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    rows: Sequence[CorpusRow],
) -> tuple[dict[str, Any], list[int]]:
    captures: dict[str, Any] = {}
    depths = coarse_depths(bundle.anatomy.n_layers, str(ctx.args.prompt_set))
    for i, row in enumerate(rows, start=1):
        captures[row.row_id] = bench.run_with_residual_cache(bundle, row.text)
        if i % max(1, len(rows) // 4) == 0:
            print(f"[lab30] captured {i}/{len(rows)} corpus rows")
    return captures, depths


def train_rows(rows: Sequence[CorpusRow]) -> list[CorpusRow]:
    selected = [r for r in rows if r.split == "train"]
    return selected if selected else list(rows)


def direction_for_domain(
    rows: Sequence[CorpusRow],
    captures: Mapping[str, Any],
    domain: str,
    depth: int,
) -> Any:
    import torch

    pos_vecs: list[Any] = []
    neg_vecs: list[Any] = []
    for row in rows:
        vec = captures[row.row_id].streams[depth, -1].float().cpu()
        if row.domain == domain:
            pos_vecs.append(vec)
        else:
            neg_vecs.append(vec)
    if not pos_vecs or not neg_vecs:
        raise ValueError(f"Cannot build direction for domain {domain} at depth {depth}")
    direction = torch.stack(pos_vecs).mean(dim=0) - torch.stack(neg_vecs).mean(dim=0)
    return direction


def score_direction(captures: Mapping[str, Any], row: CorpusRow, depth: int, direction: Any) -> float:
    return float(captures[row.row_id].streams[depth, -1].float().cpu().dot(direction.float().cpu()))


def top_contexts(rows: Sequence[CorpusRow], scores: Mapping[str, float], k: int = 5) -> list[str]:
    return [r.row_id for r in sorted(rows, key=lambda row: scores[row.row_id], reverse=True)[:k]]


def deterministic_random_like(vector: Any, key: str) -> Any:
    import torch

    gen = torch.Generator(device="cpu").manual_seed(stable_int(key) % (2**31 - 1))
    rand = torch.randn(vector.shape, generator=gen, dtype=vector.dtype)
    return rand / rand.float().norm().clamp_min(1e-8) * vector.float().norm().clamp_min(1e-8)


def build_nodes(
    ctx: bench.RunContext,
    rows: Sequence[CorpusRow],
    captures: Mapping[str, Any],
    depths: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    domains = sorted({r.domain for r in rows})
    fit_rows = train_rows(rows)
    nodes: list[dict[str, Any]] = []
    state: dict[tuple[str, int], dict[str, Any]] = {}
    for depth in depths:
        for domain in domains:
            direction = direction_for_domain(fit_rows, captures, domain, depth)
            direction_norm = float(direction.float().norm())
            all_scores = {r.row_id: score_direction(captures, r, depth, direction) for r in rows}
            random_direction = deterministic_random_like(direction, f"{domain}|{depth}|node")
            random_scores = {r.row_id: score_direction(captures, r, depth, random_direction) for r in rows}
            labels = [1 if r.domain == domain else 0 for r in rows]
            eval_rows = [r for r in rows if r.split in {"eval", "heldout"}]
            eval_labels = [1 if r.domain == domain else 0 for r in eval_rows]
            eval_scores = [all_scores[r.row_id] for r in eval_rows]
            train_auc = auc_binary(labels, [all_scores[r.row_id] for r in rows])
            eval_auc = auc_binary(eval_labels, eval_scores)
            random_auc = auc_binary(labels, [random_scores[r.row_id] for r in rows])
            top = top_contexts(rows, all_scores, k=5)
            node_id = f"{domain}@d{depth}"
            nodes.append({
                "node_id": node_id,
                "model": "loaded_model",
                "domain": domain,
                "depth": depth,
                "feature_kind": "supervised_prototype_direction",
                "direction_norm": rounded(direction_norm),
                "train_auc": rounded(train_auc),
                "eval_auc": rounded(eval_auc),
                "random_direction_auc": rounded(random_auc),
                "auc_lift_over_random": rounded(train_auc - random_auc),
                "top_contexts": " ".join(top),
                "label": domain,
                "n_positive_rows": sum(1 for r in rows if r.domain == domain),
                "n_rows": len(rows),
            })
            state[(domain, depth)] = {
                "direction": direction.float().cpu(),
                "scores": all_scores,
                "random_scores": random_scores,
                "top_contexts": top,
                "train_auc": train_auc,
                "eval_auc": eval_auc,
                "random_auc": random_auc,
            }
    path = ctx.path("tables", "feature_lineage_nodes.csv")
    bench.write_csv_with_context(ctx, path, nodes)
    ctx.register_artifact(path, "table", "Layerwise supervised feature-direction nodes.")
    return nodes, state


def lineage_score(cos_value: float, corr_value: float, top_jaccard: float, auc_a: float, auc_b: float) -> float:
    cos_part = (cos_value + 1.0) / 2.0 if math.isfinite(cos_value) else 0.0
    corr_part = (corr_value + 1.0) / 2.0 if math.isfinite(corr_value) else 0.0
    auc_part = max(0.0, min(1.0, safe_mean([auc_a, auc_b], default=0.0)))
    return 0.35 * cos_part + 0.30 * corr_part + 0.20 * top_jaccard + 0.15 * auc_part


def build_edges(
    ctx: bench.RunContext,
    rows: Sequence[CorpusRow],
    depths: Sequence[int],
    node_state: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    domains = sorted({r.domain for r in rows})
    edge_rows: list[dict[str, Any]] = []
    for d0, d1 in zip(depths, depths[1:]):
        for source_domain in domains:
            source = node_state[(source_domain, d0)]
            for target_domain in domains:
                target = node_state[(target_domain, d1)]
                cos_value = cosine(source["direction"], target["direction"])
                corr_value = safe_corr([source["scores"][r.row_id] for r in rows], [target["scores"][r.row_id] for r in rows])
                top_j = jaccard(source["top_contexts"], target["top_contexts"])
                score = lineage_score(cos_value, corr_value, top_j, float(source["eval_auc"]), float(target["eval_auc"]))
                random_target = node_state[(target_domain, d1)]["direction"]
                rand_source = deterministic_random_like(source["direction"], f"{source_domain}|{target_domain}|{d0}|{d1}|edge")
                rand_cos = cosine(rand_source, random_target)
                rand_corr = safe_corr(
                    [float(capt_score) for capt_score in source["random_scores"].values()],
                    [target["scores"][r.row_id] for r in rows],
                )
                rand_score = lineage_score(rand_cos, rand_corr, 0.0, float(source["random_auc"]), float(target["eval_auc"]))
                edge_rows.append({
                    "edge_id": f"{source_domain}@d{d0}->{target_domain}@d{d1}",
                    "source_node": f"{source_domain}@d{d0}",
                    "target_node": f"{target_domain}@d{d1}",
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "source_depth": d0,
                    "target_depth": d1,
                    "same_label": source_domain == target_domain,
                    "decoder_cosine_proxy": rounded(cos_value),
                    "activation_correlation": rounded(corr_value),
                    "top_context_jaccard": rounded(top_j),
                    "source_eval_auc": rounded(source["eval_auc"]),
                    "target_eval_auc": rounded(target["eval_auc"]),
                    "lineage_score": rounded(score),
                    "random_control_score": rounded(rand_score),
                    "lineage_lift": rounded(score - rand_score),
                    "claim_candidate": source_domain == target_domain and score >= LINEAGE_PASS_SCORE and (score - rand_score) >= LINEAGE_PASS_GAP,
                })
    path = ctx.path("tables", "feature_lineage_edges.csv")
    bench.write_csv_with_context(ctx, path, edge_rows)
    ctx.register_artifact(path, "table", "Candidate lineage edges across adjacent residual depths.")
    return edge_rows


def split_merge_tables(
    ctx: bench.RunContext,
    edge_rows: Sequence[Mapping[str, Any]],
    depths: Sequence[int],
    domains: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d0, d1 in zip(depths, depths[1:]):
        for source_domain in domains:
            outgoing = [
                r for r in edge_rows
                if r["source_domain"] == source_domain and int(r["source_depth"]) == d0 and int(r["target_depth"]) == d1
            ]
            top = sorted(outgoing, key=lambda r: float(r["lineage_score"]), reverse=True)[:3]
            scores = [max(1e-8, float(r["lineage_score"])) for r in top]
            total = sum(scores)
            entropy = -sum((s / total) * math.log(s / total) for s in scores) / math.log(len(scores)) if len(scores) > 1 else 0.0
            rows.append({
                "kind": "split_or_label_change",
                "source_depth": d0,
                "target_depth": d1,
                "source_domain": source_domain,
                "top_targets": " ".join(r["target_domain"] for r in top),
                "top_scores": " ".join(str(r["lineage_score"]) for r in top),
                "split_entropy": rounded(entropy),
                "candidate_status": "split_candidate" if entropy > 0.85 else "label_change_candidate" if top and top[0]["target_domain"] != source_domain else "single_lineage",
            })
        for target_domain in domains:
            incoming = [
                r for r in edge_rows
                if r["target_domain"] == target_domain and int(r["source_depth"]) == d0 and int(r["target_depth"]) == d1
            ]
            top = sorted(incoming, key=lambda r: float(r["lineage_score"]), reverse=True)[:3]
            scores = [max(1e-8, float(r["lineage_score"])) for r in top]
            total = sum(scores)
            entropy = -sum((s / total) * math.log(s / total) for s in scores) / math.log(len(scores)) if len(scores) > 1 else 0.0
            rows.append({
                "kind": "merge",
                "source_depth": d0,
                "target_depth": d1,
                "target_domain": target_domain,
                "top_sources": " ".join(r["source_domain"] for r in top),
                "top_scores": " ".join(str(r["lineage_score"]) for r in top),
                "merge_entropy": rounded(entropy),
                "candidate_status": "merge_candidate" if entropy > 0.85 else "single_source",
            })
    path = ctx.path("tables", "split_merge_candidates.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Possible split, merge, and label-change lineage cases.")
    return rows


def run_with_residual_addition(bundle: bench.ModelBundle, prompt: str, depth: int, vector: Any, scale: float) -> Any:
    n_layers = bundle.anatomy.n_layers
    module = bundle.final_norm if depth == n_layers else bundle.blocks[depth]

    def add_hook(mod: Any, hook_args: tuple) -> Any:
        hidden = hook_args[0].clone()
        vec = vector.float()
        vec = vec / vec.norm().clamp_min(1e-8) * float(scale)
        hidden[0, -1] = hidden[0, -1] + vec.to(hidden.device, hidden.dtype)
        return (hidden,) + tuple(hook_args[1:])

    return bench._forward_logits(bundle, prompt, [(module, add_hook)])


def causal_transfer(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    rows: Sequence[CorpusRow],
    depths: Sequence[int],
    node_state: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    domains = sorted({r.domain for r in rows})
    tok = bundle.tokenizer
    out: list[dict[str, Any]] = []
    for domain in domains:
        domain_rows = [r for r in rows if r.domain == domain]
        marker = str(domain_rows[0].labels["marker_token"])
        contrast = str(domain_rows[0].labels["contrast_token"])
        marker_id = token_ids(tok, marker)[0]
        contrast_id = token_ids(tok, contrast)[0]
        base_logits = bench.run_with_residual_cache(bundle, NEUTRAL_PROMPT).final_logits_last
        base_margin = float(base_logits[marker_id] - base_logits[contrast_id])
        for depth in depths:
            direction = node_state[(domain, depth)]["direction"]
            rand = deterministic_random_like(direction, f"{domain}|{depth}|causal")
            edited = run_with_residual_addition(bundle, NEUTRAL_PROMPT, depth, direction, CAUSAL_SCALE)
            random_logits = run_with_residual_addition(bundle, NEUTRAL_PROMPT, depth, rand, CAUSAL_SCALE)
            edited_margin = float(edited[marker_id] - edited[contrast_id])
            random_margin = float(random_logits[marker_id] - random_logits[contrast_id])
            out.append({
                "domain": domain,
                "depth": depth,
                "marker_token": marker,
                "contrast_token": contrast,
                "base_marker_minus_contrast": rounded(base_margin),
                "edited_marker_minus_contrast": rounded(edited_margin),
                "random_marker_minus_contrast": rounded(random_margin),
                "transfer_gain": rounded(edited_margin - base_margin),
                "random_gain": rounded(random_margin - base_margin),
                "control_gap": rounded((edited_margin - base_margin) - (random_margin - base_margin)),
            })
    path = ctx.path("tables", "causal_transfer_by_layer.csv")
    bench.write_csv_with_context(ctx, path, out)
    ctx.register_artifact(path, "table", "Activation-addition transfer from domain directions to marker-token logits.")
    return out


def label_stability(
    ctx: bench.RunContext,
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    domains = sorted({r["domain"] for r in nodes})
    summary: list[dict[str, Any]] = []
    overlap: list[dict[str, Any]] = []
    for domain in domains:
        nrows = [r for r in nodes if r["domain"] == domain]
        same_edges = [r for r in edges if r["source_domain"] == domain and r["target_domain"] == domain]
        all_out = [r for r in edges if r["source_domain"] == domain]
        best_node = max(nrows, key=lambda r: float(r["eval_auc"]))
        stable_edges = [r for r in same_edges if r["claim_candidate"]]
        top_same_count = 0
        for edge in same_edges:
            competitors = [
                r for r in all_out
                if int(r["source_depth"]) == int(edge["source_depth"]) and int(r["target_depth"]) == int(edge["target_depth"])
            ]
            best = max(competitors, key=lambda r: float(r["lineage_score"]))
            if best["target_domain"] == domain:
                top_same_count += 1
        transfer = [r for r in transfer_rows if r["domain"] == domain]
        summary.append({
            "domain": domain,
            "best_depth": best_node["depth"],
            "best_eval_auc": best_node["eval_auc"],
            "mean_eval_auc": rounded(safe_mean([r["eval_auc"] for r in nrows])),
            "mean_same_label_lineage_score": rounded(safe_mean([r["lineage_score"] for r in same_edges])),
            "stable_edge_count": len(stable_edges),
            "label_survival_rate": rounded(top_same_count / len(same_edges) if same_edges else 0.0),
            "mean_top_context_jaccard": rounded(safe_mean([r["top_context_jaccard"] for r in same_edges])),
            "best_causal_transfer_gap": rounded(max((float(r["control_gap"]) for r in transfer), default=float("nan"))),
            "claim_posture": "recurring_lineage_supported" if len(stable_edges) >= max(1, len(same_edges) // 2) else "lineage_needs_controls_or_refinement",
        })
        random_score = safe_mean([r["random_control_score"] for r in same_edges])
        same_score = safe_mean([r["lineage_score"] for r in same_edges])
        overlap.append({
            "domain": domain,
            "model_a": "loaded_model",
            "model_b": "same_model_cross_layer",
            "control_model": "deterministic_random_direction",
            "same_model_overlap_score": rounded(same_score),
            "random_control_overlap_score": rounded(random_score),
            "overlap_lift": rounded(same_score - random_score),
            "external_cross_model_status": "not_run_in_first_pass",
        })
    path = ctx.path("tables", "label_stability_summary.csv")
    bench.write_csv_with_context(ctx, path, summary)
    ctx.register_artifact(path, "table", "Domain label stability and lineage verdicts.")
    overlap_path = ctx.path("tables", "cross_model_feature_overlap.csv")
    bench.write_csv_with_context(ctx, overlap_path, overlap)
    ctx.register_artifact(overlap_path, "table", "Same-model lineage overlap versus random-direction control; external cross-model extension not run.")
    metrics = {
        "n_domains": len(domains),
        "supported_domains": sum(1 for r in summary if r["claim_posture"] == "recurring_lineage_supported"),
        "mean_same_model_overlap_lift": rounded(safe_mean([r["overlap_lift"] for r in overlap])),
        "mean_eval_auc": rounded(safe_mean([r["mean_eval_auc"] for r in summary])),
    }
    return summary, overlap, metrics


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
        "domains": sorted({r.domain for r in rows}),
        "directions": {f"{domain}@d{depth}": state["direction"] for (domain, depth), state in node_state.items()},
        "note": "This is a lightweight prototype dictionary, not an SAE/crosscoder dictionary.",
    }
    path = ctx.path("state", "cross_layer_dictionary.pt")
    torch.save(dictionary, path)
    ctx.register_artifact(path, "state", "Prototype cross-layer feature dictionary.")
    graph = {
        "nodes": list(nodes),
        "edges": [r for r in edges if r.get("claim_candidate") or r.get("same_label")],
        "non_claim": "Graph edges are candidate recurring directions, not proof of feature identity.",
    }
    graph_path = ctx.path("state", "lineage_graph.json")
    bench.write_json(graph_path, graph)
    ctx.register_artifact(graph_path, "state", "Lineage graph JSON for candidate recurring directions.")


def write_method_card(ctx: bench.RunContext, bundle: bench.ModelBundle, summary: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Lab 30 method card",
        "",
        "This lab uses supervised prototype directions. It does not train an SAE, transcoder, or sparse crosscoder.",
        "",
        f"- model: `{bundle.anatomy.model_id}`",
        "- feature unit: domain mean-minus-rest direction at a residual depth",
        "- lineage edge: adjacent-depth direction similarity, activation correlation, top-context overlap, and label AUC",
        "- controls: deterministic random directions and confusable-domain competitors",
        "- cross-model status: first pass exports same-model overlap versus random controls; external cross-model comparison is not run",
        "- evidence rung: `DECODE + ATTR`, with a scoped activation-addition transfer probe",
        "- forbidden claim: this is the same concept everywhere in the model",
        "",
        "| domain | best depth | mean AUC | stable edges | survival | posture |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['domain']} | {row['best_depth']} | {row['mean_eval_auc']} | "
            f"{row['stable_edge_count']} | {row['label_survival_rate']} | {row['claim_posture']} |"
        )
    path = ctx.path("method_card.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 30 method card and lineage verdicts.")


def write_operationalization_audit(ctx: bench.RunContext, summary: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Lab 30 operationalization audit",
        "",
        "Favorite interpretation under attack: a feature keeps the same identity as it moves across layers.",
        "",
        "## What the measurement can say",
        "",
        "A supervised domain direction recurs across adjacent residual depths with stronger similarity, activation correlation, and top-context overlap than random-direction controls.",
        "",
        "## What it cannot say",
        "",
        "It cannot identify an SAE feature, prove monosemanticity, or show that a concept is identical across layers or models.",
        "",
        "## Cheap explanations",
        "",
        "- The direction tracks a surface token rather than a domain feature.",
        "- Confusable domains share vocabulary and create false lineage edges.",
        "- A high cosine appears because residual streams are globally aligned.",
        "- Top-context overlap is driven by duplicated prompt templates.",
        "- The activation addition changes marker logits without preserving semantics.",
        "",
        "## Verdicts",
        "",
    ]
    for row in summary:
        lines.append(f"- `{row['domain']}`: `{row['claim_posture']}` with label survival `{row['label_survival_rate']}`.")
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Controls and non-claims for Lab 30 feature lineage.")


def write_run_summary(
    ctx: bench.RunContext,
    data_info: Mapping[str, Any],
    metrics: Mapping[str, Any],
    summary: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Lab 30 run summary: cross-layer and cross-model feature geometry",
        "",
        f"- data rows: {data_info['n_rows_selected']} selected from `{pathlib.Path(str(data_info['data_path'])).name}`",
        f"- science scope: {data_info['science_scope']}",
        f"- domains: `{data_info['domains']}`",
        f"- supported domains: `{metrics['supported_domains']}` / `{metrics['n_domains']}`",
        "",
        "## Headline verdicts",
        "",
        "| domain | best depth | mean AUC | best transfer gap | posture |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(f"| `{row['domain']}` | {row['best_depth']} | {row['mean_eval_auc']} | {row['best_causal_transfer_gap']} | {row['claim_posture']} |")
    lines += [
        "",
        "## Reading order",
        "",
        "1. `method_card.md` for the first-pass prototype-dictionary scope.",
        "2. `tables/feature_lineage_nodes.csv` for layerwise domain direction quality.",
        "3. `tables/feature_lineage_edges.csv` for adjacent-depth lineage candidates and controls.",
        "4. `tables/split_merge_candidates.csv` before claiming a split or merge.",
        "5. `tables/cross_model_feature_overlap.csv` for the honest first-pass cross-model limitation.",
        "6. `operationalization_audit.md` before writing a feature-identity claim.",
        "",
        "## Smallest surviving claim",
        "",
        "A supported row means a supervised domain direction recurred across several adjacent depths above random controls. It does not mean the same monosemantic feature exists everywhere.",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 30 run summary and reading order.")


def write_plot_guide(ctx: bench.RunContext) -> None:
    rows = [
        {"plot": "plots/feature_lineage_dashboard.png", "read_for": "Domain verdicts, AUCs, and control lifts.", "non_claim": "Not SAE feature identity."},
        {"plot": "plots/cross_layer_feature_graph.png", "read_for": "Candidate same-label graph edges.", "non_claim": "Graph edges are hypotheses."},
        {"plot": "plots/lineage_similarity_matrix.png", "read_for": "Cross-domain lineage scores.", "non_claim": "High scores need confusable-domain review."},
        {"plot": "plots/feature_split_merge_atlas.png", "read_for": "Split/merge entropy patterns.", "non_claim": "Entropy is a screen, not proof."},
        {"plot": "plots/label_stability_ladder.png", "read_for": "Label survival by domain.", "non_claim": "Label survival is not semantic identity."},
        {"plot": "plots/cross_model_feature_overlap.png", "read_for": "Same-model overlap versus random controls.", "non_claim": "External cross-model comparison is not run."},
        {"plot": "plots/causal_transfer_by_layer.png", "read_for": "Activation-addition marker transfer by layer.", "non_claim": "Marker logits are a narrow causal probe."},
    ]
    path = ctx.path("plots", "plot_reading_guide.csv")
    bench.write_csv(path, rows)
    ctx.register_artifact(path, "table", "Plot reading guide for Lab 30.")


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

    domains = [r["domain"] for r in summary]
    x = np.arange(len(domains))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Lab 30 feature lineage dashboard", fontsize=14, fontweight="bold")
    axes[0, 0].bar(x, [float(r["mean_eval_auc"]) for r in summary], color="#0072B2")
    axes[0, 0].axhline(0.7, color="#444444", linestyle="--", linewidth=0.8)
    axes[0, 0].set_xticks(x, domains, rotation=35, ha="right")
    axes[0, 0].set_title("Mean eval AUC")
    axes[0, 1].bar(x, [float(r["label_survival_rate"]) for r in summary], color="#009E73")
    axes[0, 1].set_xticks(x, domains, rotation=35, ha="right")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_title("Label survival")
    axes[1, 0].bar(x, [float(r["stable_edge_count"]) for r in summary], color="#CC79A7")
    axes[1, 0].set_xticks(x, domains, rotation=35, ha="right")
    axes[1, 0].set_title("Stable edge count")
    axes[1, 1].bar(x, [float(r["best_causal_transfer_gap"]) for r in summary], color="#D55E00")
    axes[1, 1].axhline(0, color="#444444", linewidth=0.8)
    axes[1, 1].set_xticks(x, domains, rotation=35, ha="right")
    axes[1, 1].set_title("Best marker-transfer gap")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    bench.save_figure(ctx, fig, "feature_lineage_dashboard.png", "Lab 30 feature lineage dashboard.")

    same_edges = [r for r in edges if r["same_label"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for domain in domains:
        rows = [r for r in same_edges if r["source_domain"] == domain]
        ax.plot([int(r["source_depth"]) for r in rows], [float(r["lineage_score"]) for r in rows], marker="o", label=domain)
    ax.axhline(LINEAGE_PASS_SCORE, color="#444444", linestyle="--", linewidth=0.8)
    ax.set_xlabel("source depth")
    ax.set_ylabel("same-label lineage score")
    ax.set_title("Cross-layer feature graph")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "cross_layer_feature_graph.png", "Same-label lineage graph summary.")

    domain_pairs = sorted({(r["source_domain"], r["target_domain"]) for r in edges})
    mat_domains = domains
    mat = np.zeros((len(mat_domains), len(mat_domains)))
    for i, sd in enumerate(mat_domains):
        for j, td in enumerate(mat_domains):
            vals = [float(r["lineage_score"]) for r in edges if r["source_domain"] == sd and r["target_domain"] == td]
            mat[i, j] = safe_mean(vals, default=0.0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(mat_domains)), mat_domains, rotation=35, ha="right")
    ax.set_yticks(range(len(mat_domains)), mat_domains)
    ax.set_title("Lineage similarity matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "lineage_similarity_matrix.png", "Mean cross-domain lineage score matrix.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    split_vals = [float(r.get("split_entropy", 0) or 0) for r in split_rows if r["kind"] == "split_or_label_change"]
    merge_vals = [float(r.get("merge_entropy", 0) or 0) for r in split_rows if r["kind"] == "merge"]
    ax.hist(split_vals, bins=8, alpha=0.7, label="split entropy", color="#0072B2")
    ax.hist(merge_vals, bins=8, alpha=0.7, label="merge entropy", color="#D55E00")
    ax.set_title("Feature split/merge atlas")
    ax.set_xlabel("entropy")
    ax.legend(frameon=False)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "feature_split_merge_atlas.png", "Split/merge entropy atlas.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(domains, [float(r["label_survival_rate"]) for r in summary], color="#009E73")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("label survival rate")
    ax.set_title("Label stability ladder")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "label_stability_ladder.png", "Label stability ladder.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, [float(r["same_model_overlap_score"]) for r in overlap], 0.4, label="same-model", color="#0072B2")
    ax.bar(x + 0.2, [float(r["random_control_overlap_score"]) for r in overlap], 0.4, label="random control", color="#999999")
    ax.set_xticks(x, domains, rotation=35, ha="right")
    ax.set_title("Cross-model feature overlap placeholder")
    ax.legend(frameon=False)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "cross_model_feature_overlap.png", "Same-model overlap versus random controls.")

    depths = sorted({int(r["depth"]) for r in transfer_rows})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for domain in domains:
        rows = [r for r in transfer_rows if r["domain"] == domain]
        ax.plot([int(r["depth"]) for r in rows], [float(r["control_gap"]) for r in rows], marker="o", label=domain)
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xticks(depths)
    ax.set_xlabel("depth")
    ax.set_ylabel("transfer control gap")
    ax.set_title("Causal transfer by layer")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "causal_transfer_by_layer.png", "Activation-addition marker transfer by layer.")


def write_claims(ctx: bench.RunContext, summary: Sequence[Mapping[str, Any]]) -> None:
    run_name = ctx.run_dir.name
    claims = []
    for i, row in enumerate(summary, start=1):
        claims.append({
            "id": f"{LAB_ID}-C{i}",
            "tag": "DECODE,ATTR",
            "text": (
                f"For domain `{row['domain']}`, supervised prototype directions had mean eval AUC "
                f"{row['mean_eval_auc']}, stable edge count {row['stable_edge_count']}, "
                f"and label survival {row['label_survival_rate']}. Posture: {row['claim_posture']}."
            ),
            "artifact": f"runs/{run_name}/tables/label_stability_summary.csv",
            "falsifier": "Random directions, confusable domains, or held-out contexts match the same lineage score.",
        })
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    rows, data_info = load_rows(ctx)
    rows = tokenization_gate(ctx, bundle, rows)
    manifest_path = ctx.path("diagnostics", "data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Lab 30 data manifest and first-pass method scope.")
    bench.run_hook_parity_check(ctx, bundle, rows[0].text)
    first = bench.run_with_residual_cache(bundle, rows[0].text)
    bench.run_lens_self_check(ctx, bundle, first)
    bench.run_patch_noop_check(ctx, bundle, rows[0].text)
    captures, depths = capture_corpus(ctx, bundle, rows)
    nodes, node_state = build_nodes(ctx, rows, captures, depths)
    edges = build_edges(ctx, rows, depths, node_state)
    domains = sorted({r.domain for r in rows})
    split_rows = split_merge_tables(ctx, edges, depths, domains)
    transfer_rows = causal_transfer(ctx, bundle, rows, depths, node_state)
    summary, overlap, metrics = label_stability(ctx, nodes, edges, transfer_rows)
    save_state(ctx, bundle, rows, depths, node_state, nodes, edges)
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, {**metrics, "data": data_info, "depths": depths})
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 30 metrics.")
    write_method_card(ctx, bundle, summary)
    write_operationalization_audit(ctx, summary)
    write_run_summary(ctx, data_info, metrics, summary)
    write_claims(ctx, summary)
    write_plots(ctx, summary, nodes, edges, split_rows, overlap, transfer_rows)
    print(f"[lab30] wrote {len(nodes)} nodes, {len(edges)} edges, and {len(summary)} domain verdicts")
