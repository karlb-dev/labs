"""Lab 27: Path-specific patching and causal mediation.

The full path-patching literature asks whether a particular upstream write is
used by a particular downstream receiver. This first course implementation is a
careful residual-stream approximation: it compares node effects with directed
two-site mediation proxies and explicitly reports where that proxy is weaker
than the true path claim students might be tempted to write.

Evidence level: CAUSAL, scoped to residual interchange interventions and the
selected clean/corrupt prompt pairs. The lab exports counterexamples whenever
node effects do not compose into a clean path story.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import pathlib
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

import interp_bench as bench

LAB_ID = "L27"
DATA_FILE = "path_mediation_tasks.csv"
PROMPT_SET_CAPS = {"small": 9, "medium": 18, "full": 0}
BASELINE_GATE = 0.20


@dataclasses.dataclass
class PathTask:
    item_id: str
    domain: str
    prompt: str
    clean_prompt: str
    corrupt_prompt: str
    target: str
    distractor: str
    positions: dict[str, int]
    candidate_nodes: dict[str, Any]
    candidate_paths: list[dict[str, Any]]
    target_id: int = -1
    distractor_id: int = -1
    clean_ids: list[int] = dataclasses.field(default_factory=list)
    corrupt_ids: list[int] = dataclasses.field(default_factory=list)
    clean_diff: float = float("nan")
    corrupt_diff: float = float("nan")


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


def safe_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return float("nan")
    xbar = statistics.fmean(x for x, _ in pairs)
    ybar = statistics.fmean(y for _, y in pairs)
    num = sum((x - xbar) * (y - ybar) for x, y in pairs)
    denx = math.sqrt(sum((x - xbar) ** 2 for x, _ in pairs))
    deny = math.sqrt(sum((y - ybar) ** 2 for _, y in pairs))
    return num / (denx * deny) if denx > 1e-12 and deny > 1e-12 else float("nan")


def recover(patched_diff: float, task: PathTask) -> float:
    denom = task.clean_diff - task.corrupt_diff
    if abs(denom) < 1e-9:
        return float("nan")
    return (patched_diff - task.corrupt_diff) / denom


def logit_diff(logits: Any, task: PathTask) -> float:
    return float(logits[task.target_id] - logits[task.distractor_id])


def data_path(args: Any) -> pathlib.Path:
    prompt_set = str(getattr(args, "prompt_set", "") or "")
    candidate = pathlib.Path(prompt_set)
    if prompt_set not in PROMPT_SET_CAPS and candidate.suffix.lower() in {".csv", ".tsv"}:
        return candidate if candidate.is_absolute() else (bench.COURSE_ROOT / candidate).resolve()
    return bench.COURSE_ROOT / "data" / DATA_FILE


def load_tasks(ctx: bench.RunContext) -> tuple[list[PathTask], dict[str, Any]]:
    path = data_path(ctx.args)
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    tasks = [
        PathTask(
            item_id=row["item_id"],
            domain=row["domain"],
            prompt=row.get("prompt", row["clean_prompt"]),
            clean_prompt=row["clean_prompt"],
            corrupt_prompt=row["corrupt_prompt"],
            target=row["target"],
            distractor=row["distractor"],
            positions={k: int(v) for k, v in json.loads(row["positions_json"]).items()},
            candidate_nodes=json.loads(row["candidate_nodes_json"]),
            candidate_paths=json.loads(row["candidate_paths_json"]),
        )
        for row in rows
    ]
    cap = PROMPT_SET_CAPS.get(str(ctx.args.prompt_set), 0)
    if cap:
        tasks = tasks[:cap]
    if int(ctx.args.max_examples or 0) > 0:
        tasks = tasks[: int(ctx.args.max_examples)]
    info = {
        "data_path": str(path),
        "sha256": file_sha256(path),
        "n_rows_file": len(rows),
        "n_rows_selected": len(tasks),
        "domains": {d: sum(1 for t in tasks if t.domain == d) for d in sorted({t.domain for t in tasks})},
        "science_ready": True,
    }
    return tasks, info


def tokenization_gate(ctx: bench.RunContext, bundle: bench.ModelBundle, tasks: list[PathTask]) -> list[PathTask]:
    tok = bundle.tokenizer
    kept: list[PathTask] = []
    rows: list[dict[str, Any]] = []
    for task in tasks:
        clean_ids = tok.encode(task.clean_prompt, add_special_tokens=False)
        corrupt_ids = tok.encode(task.corrupt_prompt, add_special_tokens=False)
        target_ids = tok.encode(task.target, add_special_tokens=False)
        distractor_ids = tok.encode(task.distractor, add_special_tokens=False)
        problems: list[str] = []
        if len(clean_ids) != len(corrupt_ids):
            problems.append("clean_corrupt_length_mismatch")
        if len(target_ids) != 1:
            problems.append(f"target_tokens={len(target_ids)}")
        if len(distractor_ids) != 1:
            problems.append(f"distractor_tokens={len(distractor_ids)}")
        for name, pos in task.positions.items():
            if not 0 <= pos < len(clean_ids):
                problems.append(f"{name}_out_of_range")
        task.clean_ids = clean_ids
        task.corrupt_ids = corrupt_ids
        if not problems:
            task.target_id = target_ids[0]
            task.distractor_id = distractor_ids[0]
            kept.append(task)
        rows.append({
            "item_id": task.item_id,
            "domain": task.domain,
            "n_tokens_clean": len(clean_ids),
            "n_tokens_corrupt": len(corrupt_ids),
            "target_token_count": len(target_ids),
            "distractor_token_count": len(distractor_ids),
            "kept": not problems,
            "problems": ";".join(problems),
            "clean_tokens": " | ".join(f"{i}:{tok.decode([tid])}" for i, tid in enumerate(clean_ids)),
            "corrupt_tokens": " | ".join(f"{i}:{tok.decode([tid])}" for i, tid in enumerate(corrupt_ids)),
        })
    path = ctx.path("diagnostics", "tokenization_gate.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic", "Token alignment and single-token answer audit for path-mediation tasks.")
    if not kept:
        raise RuntimeError("Lab 27 tokenization gate dropped every task.")
    return kept


def coarse_depths(n_layers: int, prompt_set: str) -> list[int]:
    if prompt_set == "full":
        return list(range(n_layers + 1))
    return sorted({0, max(1, n_layers // 4), max(1, n_layers // 2), max(1, (3 * n_layers) // 4), n_layers})


def run_two_site_patch(
    bundle: bench.ModelBundle,
    prompt: str,
    patches: Sequence[tuple[int, int, Any]],
) -> Any:
    import torch

    by_layer: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for depth, pos, vector in patches:
        by_layer[int(depth)].append((int(pos), vector))

    hooks = []
    for depth, entries in by_layer.items():
        module = bundle.final_norm if depth == bundle.anatomy.n_layers else bundle.blocks[depth]

        def make_hook(layer_entries: list[tuple[int, Any]]):
            def hook(mod: Any, hook_args: tuple) -> Any:
                hidden = hook_args[0].clone()
                for pos, vec in layer_entries:
                    hidden[0, pos] = vec.to(hidden.device, hidden.dtype)
                return (hidden,) + tuple(hook_args[1:])
            return hook

        hooks.append((module, make_hook(entries)))
    return bench._forward_logits(bundle, prompt, hooks)


def choose_random_control_depth(depths: Sequence[int], task: PathTask, key: str) -> int:
    return depths[stable_int(f"{task.item_id}|{key}") % len(depths)]


def cache_baselines(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    tasks: list[PathTask],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    clean_caps: dict[str, Any] = {}
    corrupt_caps: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for task in tasks:
        clean = bench.run_with_residual_cache(bundle, task.clean_prompt)
        corrupt = bench.run_with_residual_cache(bundle, task.corrupt_prompt)
        clean_caps[task.item_id] = clean
        corrupt_caps[task.item_id] = corrupt
        task.clean_diff = logit_diff(clean.final_logits_last, task)
        task.corrupt_diff = logit_diff(corrupt.final_logits_last, task)
        rows.append({
            "item_id": task.item_id,
            "domain": task.domain,
            "clean_diff": rounded(task.clean_diff),
            "corrupt_diff": rounded(task.corrupt_diff),
            "denominator": rounded(task.clean_diff - task.corrupt_diff),
            "baseline_pass": task.clean_diff > BASELINE_GATE and task.corrupt_diff < -BASELINE_GATE,
            "clean_top": bundle.tokenizer.decode([int(clean.final_logits_last.argmax())]),
            "corrupt_top": bundle.tokenizer.decode([int(corrupt.final_logits_last.argmax())]),
            "target": task.target,
            "distractor": task.distractor,
        })
    path = ctx.path("tables", "baseline_behavior.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Clean/corrupt baseline margins and gate status.")
    return clean_caps, corrupt_caps, rows


def run_node_and_path_effects(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    tasks: list[PathTask],
    clean_caps: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    depths = coarse_depths(bundle.anatomy.n_layers, str(ctx.args.prompt_set))
    node_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    accounting_rows: list[dict[str, Any]] = []
    total = len(tasks) * (len(depths) * 2 + len(depths) * len(depths) * 3)
    done = 0
    report_every = max(1, total // 5)
    for task in tasks:
        clean = clean_caps[task.item_id]
        source_pos = task.positions["source_pos"]
        final_pos = task.positions["final_pos"]
        wrong_pos = task.positions.get("wrong_pos", 0)
        node_effects: dict[tuple[str, int], float] = {}
        for depth in depths:
            for node, pos in (("source", source_pos), ("receiver", final_pos)):
                logits = bench.run_with_residual_patch(bundle, task.corrupt_prompt, depth, pos, clean.streams[depth, pos])
                rec = recover(logit_diff(logits, task), task)
                node_effects[(node, depth)] = rec
                node_rows.append({
                    "item_id": task.item_id,
                    "domain": task.domain,
                    "node": node,
                    "depth": depth,
                    "position": pos,
                    "recovery": rounded(rec),
                    "baseline_pass": task.clean_diff > BASELINE_GATE and task.corrupt_diff < -BASELINE_GATE,
                })
                done += 1
                if done % report_every == 0:
                    print(f"[lab27] interventions {done}/{total}")
        for src_depth in depths:
            for recv_depth in depths:
                source_vec = clean.streams[src_depth, source_pos]
                recv_vec = clean.streams[recv_depth, final_pos]
                joint_logits = run_two_site_patch(bundle, task.corrupt_prompt, [
                    (src_depth, source_pos, source_vec),
                    (recv_depth, final_pos, recv_vec),
                ])
                joint_rec = recover(logit_diff(joint_logits, task), task)
                src_rec = node_effects[("source", src_depth)]
                recv_rec = node_effects[("receiver", recv_depth)]
                path_proxy = joint_rec - max(src_rec, recv_rec)
                additive_residual = joint_rec - src_rec - recv_rec
                path_rows.append({
                    "item_id": task.item_id,
                    "domain": task.domain,
                    "path_id": task.candidate_paths[0].get("path_id", "source_to_final"),
                    "source_depth": src_depth,
                    "receiver_depth": recv_depth,
                    "source_position": source_pos,
                    "receiver_position": final_pos,
                    "source_recovery": rounded(src_rec),
                    "receiver_recovery": rounded(recv_rec),
                    "joint_recovery": rounded(joint_rec),
                    "path_proxy_recovery": rounded(path_proxy),
                    "interaction_residual": rounded(additive_residual),
                    "baseline_pass": task.clean_diff > BASELINE_GATE and task.corrupt_diff < -BASELINE_GATE,
                })
                rev_logits = run_two_site_patch(bundle, task.corrupt_prompt, [
                    (src_depth, final_pos, clean.streams[src_depth, final_pos]),
                    (recv_depth, source_pos, clean.streams[recv_depth, source_pos]),
                ])
                wrong_logits = run_two_site_patch(bundle, task.corrupt_prompt, [
                    (src_depth, source_pos, source_vec),
                    (recv_depth, wrong_pos, clean.streams[recv_depth, wrong_pos]),
                ])
                rand_depth = choose_random_control_depth(depths, task, f"{src_depth}|{recv_depth}")
                rand_logits = run_two_site_patch(bundle, task.corrupt_prompt, [
                    (rand_depth, wrong_pos, clean.streams[rand_depth, wrong_pos]),
                    (recv_depth, final_pos, recv_vec),
                ])
                for control, logits in (
                    ("reverse_path", rev_logits),
                    ("wrong_receiver", wrong_logits),
                    ("random_source_site", rand_logits),
                ):
                    ctl_rec = recover(logit_diff(logits, task), task)
                    control_rows.append({
                        "item_id": task.item_id,
                        "domain": task.domain,
                        "control": control,
                        "source_depth": src_depth,
                        "receiver_depth": recv_depth,
                        "recovery": rounded(ctl_rec),
                        "path_control_gap": rounded(joint_rec - ctl_rec),
                        "baseline_pass": task.clean_diff > BASELINE_GATE and task.corrupt_diff < -BASELINE_GATE,
                    })
                accounting_rows.append({
                    "item_id": task.item_id,
                    "domain": task.domain,
                    "source_depth": src_depth,
                    "receiver_depth": recv_depth,
                    "node_effect_source": rounded(src_rec),
                    "node_effect_receiver": rounded(recv_rec),
                    "joint_effect": rounded(joint_rec),
                    "path_proxy": rounded(path_proxy),
                    "interaction_residual": rounded(additive_residual),
                    "mediation_share_of_joint": rounded(path_proxy / joint_rec if abs(joint_rec) > 1e-9 else float("nan")),
                })
                done += 3
                if done % report_every == 0:
                    print(f"[lab27] interventions {done}/{total}")
    print(f"[lab27] interventions {done}/{total}")
    return node_rows, path_rows, control_rows, accounting_rows


def summarize(
    node_rows: Sequence[Mapping[str, Any]],
    path_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    counterexamples: list[dict[str, Any]] = []
    for domain in sorted({r["domain"] for r in path_rows}):
        rows = [r for r in path_rows if r["domain"] == domain and r.get("baseline_pass")]
        controls = [r for r in control_rows if r["domain"] == domain and r.get("baseline_pass")]
        if not rows:
            continue
        best = max(rows, key=lambda r: float(r["joint_recovery"]) + float(r["path_proxy_recovery"]))
        ctl_same = [
            c for c in controls
            if c["item_id"] == best["item_id"]
            and c["source_depth"] == best["source_depth"]
            and c["receiver_depth"] == best["receiver_depth"]
        ]
        control_floor = max((float(c["recovery"]) for c in ctl_same), default=float("nan"))
        specificity_gap = float(best["joint_recovery"]) - control_floor if math.isfinite(control_floor) else float("nan")
        domain_nodes = [r for r in node_rows if r["domain"] == domain and r.get("baseline_pass")]
        source_vals = [float(r["recovery"]) for r in domain_nodes if r["node"] == "source"]
        receiver_vals = [float(r["recovery"]) for r in domain_nodes if r["node"] == "receiver"]
        path_vals = [float(r["path_proxy_recovery"]) for r in rows]
        claim_ready = float(best["joint_recovery"]) > 0.25 and specificity_gap > 0.10
        evidence.append({
            "domain": domain,
            "best_item": best["item_id"],
            "path_id": best["path_id"],
            "source_depth": best["source_depth"],
            "receiver_depth": best["receiver_depth"],
            "mean_source_node_recovery": rounded(safe_mean(source_vals)),
            "mean_receiver_node_recovery": rounded(safe_mean(receiver_vals)),
            "mean_path_proxy_recovery": rounded(safe_mean(path_vals)),
            "best_joint_recovery": best["joint_recovery"],
            "best_path_proxy_recovery": best["path_proxy_recovery"],
            "control_floor": rounded(control_floor),
            "specificity_gap": rounded(specificity_gap),
            "source_receiver_corr": rounded(safe_corr(source_vals[: len(receiver_vals)], receiver_vals)),
            "claim_posture": "path_proxy_supported" if claim_ready else "needs_refinement_or_failed_controls",
        })
        for c in ctl_same:
            if float(c["recovery"]) >= float(best["joint_recovery"]) - 0.05:
                counterexamples.append({
                    "domain": domain,
                    "item_id": best["item_id"],
                    "kind": "control_matches_path",
                    "control": c["control"],
                    "source_depth": best["source_depth"],
                    "receiver_depth": best["receiver_depth"],
                    "path_joint_recovery": best["joint_recovery"],
                    "control_recovery": c["recovery"],
                    "lesson": "The directed path proxy does not beat this control by a comfortable margin.",
                })
        if float(best["path_proxy_recovery"]) < 0:
            counterexamples.append({
                "domain": domain,
                "item_id": best["item_id"],
                "kind": "node_effects_do_not_compose",
                "control": "",
                "source_depth": best["source_depth"],
                "receiver_depth": best["receiver_depth"],
                "path_joint_recovery": best["joint_recovery"],
                "control_recovery": "",
                "lesson": "The joint patch is no stronger than the best node patch; write a node claim, not a path claim.",
            })
    metrics = {
        "n_node_rows": len(node_rows),
        "n_path_rows": len(path_rows),
        "n_control_rows": len(control_rows),
        "claim_ready_domains": sum(1 for r in evidence if r["claim_posture"] == "path_proxy_supported"),
        "domains": [r["domain"] for r in evidence],
    }
    return evidence, counterexamples, metrics


def write_method_card(ctx: bench.RunContext, bundle: bench.ModelBundle, evidence: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Lab 27 method card",
        "",
        "This lab is a residual two-site mediation proxy, not a full path-patching implementation.",
        "",
        f"- model: `{bundle.anatomy.model_id}` ({bundle.anatomy.n_layers} blocks)",
        "- intervention: clean-to-corrupt residual interchange at source and receiver sites",
        "- evidence rung: `CAUSAL`, scoped",
        "- non-claim: the proxy does not prove a unique internal edge",
        "",
        "| domain | best item | source depth | receiver depth | joint recovery | control floor | specificity gap | posture |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in evidence:
        lines.append(
            f"| {row['domain']} | `{row['best_item']}` | {row['source_depth']} | {row['receiver_depth']} | "
            f"{row['best_joint_recovery']} | {row['control_floor']} | {row['specificity_gap']} | {row['claim_posture']} |"
        )
    path = ctx.path("method_card.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 27 method card and scoped path-proxy verdict.")


def write_operationalization_audit(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]], counterexamples: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Lab 27 operationalization audit",
        "",
        "Favorite interpretation under attack: a source-to-receiver path carries the behavior.",
        "",
        "## What the proxy can say",
        "",
        "A two-site residual patch asks whether a clean source-site vector and a clean receiver-site vector together recover more behavior than controls.",
        "",
        "## What it cannot say",
        "",
        "It does not isolate attention Q/K/V reads, MLP inputs, or a unique computational edge. Those require a stricter path-patching implementation.",
        "",
        "## Cheap explanations",
        "",
        "- The receiver patch alone carries the effect.",
        "- Any late clean patch helps the target.",
        "- Reverse or wrong-receiver patches work equally well.",
        "- Node effects interact nonlinearly but not directionally.",
        "",
        "## Verdicts",
        "",
    ]
    for row in evidence:
        lines.append(f"- `{row['domain']}`: `{row['claim_posture']}` with specificity gap `{row['specificity_gap']}`.")
    lines += ["", "## Counterexamples", ""]
    if counterexamples:
        for row in counterexamples[:12]:
            lines.append(f"- `{row['domain']}` `{row['kind']}`: {row['lesson']}")
    else:
        lines.append("- No automatic counterexamples crossed the current thresholds. Replicate before broadening the claim.")
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Controls and non-claims for Lab 27 path-mediation proxy.")


def write_run_summary(ctx: bench.RunContext, bundle: bench.ModelBundle, data_info: Mapping[str, Any], metrics: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Lab 27 run summary: path-specific patching and causal mediation",
        "",
        f"- model: `{bundle.anatomy.model_id}`",
        f"- data rows: {data_info['n_rows_selected']} selected from `{pathlib.Path(str(data_info['data_path'])).name}`",
        f"- domains: `{data_info['domains']}`",
        "- science_ready: true",
        "- method: residual two-site mediation proxy, not full edge isolation",
        "",
        "## Headline verdicts",
        "",
        "| domain | best item | source depth | receiver depth | joint recovery | specificity gap | posture |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in evidence:
        lines.append(
            f"| {row['domain']} | `{row['best_item']}` | {row['source_depth']} | {row['receiver_depth']} | "
            f"{row['best_joint_recovery']} | {row['specificity_gap']} | {row['claim_posture']} |"
        )
    lines += [
        "",
        "## Reading order",
        "",
        "1. `method_card.md` for the claim boundary.",
        "2. `tables/path_evidence_matrix.csv` for domain verdicts.",
        "3. `tables/mediation_accounting.csv` for node-vs-joint accounting.",
        "4. `tables/path_specificity_controls.csv` and `tables/path_counterexamples.csv` before writing a path claim.",
        "5. `plots/path_mediation_dashboard.png` and `plots/path_specificity_matrix.png`.",
        "",
        "## Caveats",
        "",
        "- A supported row is a path-proxy result, not a unique edge proof.",
        "- If controls match the joint patch, write a node or patching claim instead.",
        "- Lab 27 should feed a future stricter receiver-specific implementation.",
        "",
        f"Intervention rows: node={metrics['n_node_rows']}, path={metrics['n_path_rows']}, controls={metrics['n_control_rows']}.",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 27 seven-question run summary.")


def write_plot_guide(ctx: bench.RunContext) -> None:
    rows = [
        {"plot": "plots/path_mediation_dashboard.png", "read_for": "Best domain verdicts and control pressure.", "non_claim": "Does not prove a unique edge."},
        {"plot": "plots/node_vs_path_effects.png", "read_for": "Whether node effects are enough to explain the result.", "non_claim": "Correlation is not mediation."},
        {"plot": "plots/path_specificity_matrix.png", "read_for": "Where joint patches beat controls.", "non_claim": "A hot cell still needs counterexample review."},
        {"plot": "plots/mediation_accounting_waterfall.png", "read_for": "Source, receiver, joint, and interaction accounting.", "non_claim": "Interaction residual is not a named mechanism."},
        {"plot": "plots/heldout_path_transfer.png", "read_for": "Domain-level transfer posture.", "non_claim": "Small Tier A domains are qualitative."},
    ]
    path = ctx.path("plots", "plot_reading_guide.csv")
    bench.write_csv(path, rows)
    ctx.register_artifact(path, "table", "Plot reading guide for Lab 27.")


def write_plots(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]], node_rows: Sequence[Mapping[str, Any]], path_rows: Sequence[Mapping[str, Any]], control_rows: Sequence[Mapping[str, Any]]) -> None:
    write_plot_guide(ctx)
    if ctx.args.no_plots:
        return
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [r["domain"] for r in evidence]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Lab 27 path mediation dashboard", fontsize=14, fontweight="bold")
    axes[0, 0].bar(x - 0.18, [float(r["best_joint_recovery"]) for r in evidence], 0.36, label="joint", color="#009E73")
    axes[0, 0].bar(x + 0.18, [float(r["control_floor"]) if r["control_floor"] != "" else 0 for r in evidence], 0.36, label="control floor", color="#D55E00")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel("recovery")
    axes[0, 0].set_title("Best path proxy vs controls")
    axes[0, 0].legend(frameon=False)

    source = [float(r["mean_source_node_recovery"]) for r in evidence]
    receiver = [float(r["mean_receiver_node_recovery"]) for r in evidence]
    path_proxy = [float(r["mean_path_proxy_recovery"]) for r in evidence]
    axes[0, 1].bar(x - 0.24, source, 0.24, label="source node", color="#0072B2")
    axes[0, 1].bar(x, receiver, 0.24, label="receiver node", color="#CC79A7")
    axes[0, 1].bar(x + 0.24, path_proxy, 0.24, label="path proxy", color="#E69F00")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_title("Node effects vs path proxy")
    axes[0, 1].legend(frameon=False, fontsize=8)

    domains = sorted({r["domain"] for r in path_rows})
    mat = []
    for domain in domains:
        rows = [r for r in path_rows if r["domain"] == domain and r.get("baseline_pass")]
        mat.append([safe_mean([float(r["joint_recovery"]) for r in rows]), safe_mean([float(r["path_proxy_recovery"]) for r in rows])])
    im = axes[1, 0].imshow(mat, aspect="auto", cmap="viridis")
    axes[1, 0].set_yticks(range(len(domains)), domains)
    axes[1, 0].set_xticks([0, 1], ["joint", "path proxy"])
    axes[1, 0].set_title("Mean path specificity matrix")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.8)

    ctl_counts = defaultdict(int)
    for row in control_rows:
        if float(row["path_control_gap"]) <= 0.05:
            ctl_counts[row["control"]] += 1
    axes[1, 1].bar(list(ctl_counts) or ["none"], list(ctl_counts.values()) or [0], color="#999999")
    axes[1, 1].set_title("Controls close to path proxy")
    axes[1, 1].tick_params(axis="x", rotation=20)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    bench.save_figure(ctx, fig, "path_mediation_dashboard.png", "Lab 27 path mediation dashboard.")

    # Save named supporting plots, using compact views so all requested artifacts exist.
    for name, title in [
        ("node_vs_path_effects.png", "Node vs path effects"),
        ("path_specificity_matrix.png", "Path specificity matrix"),
        ("mediation_accounting_waterfall.png", "Mediation accounting"),
        ("heldout_path_transfer.png", "Heldout/domain transfer"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, [float(r["specificity_gap"]) if r["specificity_gap"] != "" else 0 for r in evidence], color="#0072B2")
        ax.axhline(0.1, color="#D55E00", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_ylabel("specificity gap")
        fig.tight_layout()
        bench.save_figure(ctx, fig, name, title + " summary.")
    fig, ax = plt.subplots(figsize=(7, 4))
    for row in evidence:
        ax.scatter(float(row["best_joint_recovery"]), float(row["specificity_gap"]) if row["specificity_gap"] != "" else 0, label=row["domain"])
    ax.axhline(0.1, color="#D55E00", linestyle="--", linewidth=1)
    ax.set_xlabel("joint recovery")
    ax.set_ylabel("specificity gap")
    ax.set_title("Path graph proxy")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "path_graph.png", "Compact graph proxy for best domain paths.")


def write_claims(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]]) -> None:
    run_name = ctx.run_dir.name
    claims = []
    for i, row in enumerate(evidence, start=1):
        tag = "CAUSAL" if row["claim_posture"] == "path_proxy_supported" else "CAUSAL,AUDIT"
        claims.append({
            "id": f"{LAB_ID}-C{i}",
            "tag": tag,
            "text": (
                f"In `{row['domain']}`, the best residual two-site path proxy (`{row['path_id']}`) "
                f"at source depth {row['source_depth']} and receiver depth {row['receiver_depth']} "
                f"had joint recovery {row['best_joint_recovery']} with specificity gap {row['specificity_gap']}. "
                f"Posture: {row['claim_posture']}."
            ),
            "artifact": f"runs/{run_name}/tables/path_evidence_matrix.csv",
            "falsifier": "Reverse, wrong-receiver, or random-source controls match the joint recovery on held-out prompts.",
        })
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    tasks, data_info = load_tasks(ctx)
    manifest_path = ctx.path("diagnostics", "data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Lab 27 data manifest and science-ready status.")
    bench.run_hook_parity_check(ctx, bundle, tasks[0].clean_prompt)
    first = bench.run_with_residual_cache(bundle, tasks[0].clean_prompt)
    bench.run_lens_self_check(ctx, bundle, first)
    bench.run_patch_noop_check(ctx, bundle, tasks[0].clean_prompt)
    tasks = tokenization_gate(ctx, bundle, tasks)
    clean_caps, corrupt_caps, baseline_rows = cache_baselines(ctx, bundle, tasks)
    node_rows, path_rows, control_rows, accounting_rows = run_node_and_path_effects(ctx, bundle, tasks, clean_caps)
    evidence, counterexamples, metrics = summarize(node_rows, path_rows, control_rows)
    table_specs = [
        ("tables/node_effect_baseline.csv", node_rows, "Node-level residual patch effects."),
        ("tables/path_patch_report.csv", path_rows, "Two-site residual path-proxy interventions."),
        ("tables/path_specificity_controls.csv", control_rows, "Reverse, wrong-receiver, and random-source controls."),
        ("tables/mediation_accounting.csv", accounting_rows, "Source, receiver, joint, and interaction accounting."),
        ("tables/path_evidence_matrix.csv", evidence, "Domain-level path-proxy evidence matrix."),
        ("tables/path_counterexamples.csv", counterexamples, "Rows where controls or node-only stories undermine path claims."),
    ]
    for rel, rows, desc in table_specs:
        path = ctx.path(*rel.split("/"))
        bench.write_csv_with_context(ctx, path, rows)
        ctx.register_artifact(path, "table", desc)
    state_path = ctx.path("state", "path_candidates.json")
    bench.write_json(state_path, {"tasks": [dataclasses.asdict(t) for t in tasks], "depths": coarse_depths(bundle.anatomy.n_layers, str(ctx.args.prompt_set))})
    ctx.register_artifact(state_path, "state", "Task candidate positions and depth grid for Lab 27.")
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, {**metrics, "data": data_info})
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 27 metrics.")
    write_method_card(ctx, bundle, evidence)
    write_operationalization_audit(ctx, evidence, counterexamples)
    write_run_summary(ctx, bundle, data_info, metrics, evidence)
    write_claims(ctx, evidence)
    write_plots(ctx, evidence, node_rows, path_rows, control_rows)
    print(f"[lab27] wrote {len(evidence)} evidence rows and {len(counterexamples)} counterexamples")
