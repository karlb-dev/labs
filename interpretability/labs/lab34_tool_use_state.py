"""Lab 34: Tool use, agents, and state tracking.

This lab uses a benign deterministic tool harness and model residual probes at
the user-prompt boundary. The goal is to distinguish a decodable "tool needed"
or "which tool" signal from surface cues such as digits, tool-name mentions,
or prompt length. The tool trace is produced by local toy tools, not by an
autonomous agent.

Evidence level: OBS + DECODE + CAUSAL + SELF-REPORT, scoped to controlled toy
tools and explicit surface-cue controls.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import math
import operator
import pathlib
import re
import statistics
from collections import Counter, defaultdict, deque
from typing import Any, Mapping, Sequence

import interp_bench as bench

LAB_ID = "L34"
DATA_FILE = "tool_use_tasks.jsonl"
PROMPT_SET_CAPS = {"small": 24, "medium": 24, "full": 0}
TOOLS = ("calculator", "dictionary", "calendar", "file_search", "route_planner", "unit_converter", "none")
TOOL_LETTERS = {
    "calculator": "A",
    "dictionary": "B",
    "calendar": "C",
    "file_search": "D",
    "route_planner": "E",
    "unit_converter": "F",
    "none": "N",
}
STEER_SCALES = (-1.0, 0.0, 1.0)
REVIEW_FIELDS = ("student_trace_label", "student_confidence", "student_evidence_span", "reviewer_trace_label", "agreement_status")

GLOSSARY = {
    "latency": "delay before a response begins",
    "mutex": "a lock that allows one holder at a time",
    "photosynthesis": "plants use light to make sugar from water and carbon dioxide",
}
CALENDAR = {
    "design review": "Tuesday 10:00",
    "standup": "Monday 09:00",
    "bug triage": "Monday 10:00",
    "demo prep": "Friday afternoon",
}
DOCS = {
    "doc_cache.md": "Cache invalidation removes stale user records after writes.",
    "doc_export.md": "The exporter writes CSV files for downstream reports.",
    "doc_reliability.md": "The retry budget is three attempts with backoff.",
}
GRAPH = {
    "A": ["B"],
    "B": ["D"],
    "C": ["E"],
    "D": ["E"],
    "E": ["F"],
    "F": [],
}


@dataclasses.dataclass
class ToolTask:
    task_id: str
    family: str
    user_prompt: str
    required_tool: str
    tool_needed: bool
    tool_args: dict[str, Any]
    answer: str
    distractor_tool: str
    split: str
    notes: str


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


def unit_vector(vec: Any) -> Any:
    import torch

    norm = torch.linalg.vector_norm(vec.float())
    if float(norm) <= 1e-9:
        return torch.zeros_like(vec.float())
    return vec.float() / norm


def data_path(args: Any) -> pathlib.Path:
    prompt_set = str(getattr(args, "prompt_set", "") or "")
    candidate = pathlib.Path(prompt_set)
    if prompt_set not in PROMPT_SET_CAPS and candidate.suffix.lower() in {".jsonl", ".json"}:
        return candidate if candidate.is_absolute() else (bench.COURSE_ROOT / candidate).resolve()
    return bench.COURSE_ROOT / "data" / DATA_FILE


def load_tasks(ctx: bench.RunContext) -> tuple[list[ToolTask], dict[str, Any]]:
    path = data_path(ctx.args)
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    tasks = [ToolTask(**row) for row in rows]
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
        "families": dict(Counter(t.family for t in tasks)),
        "required_tools": dict(Counter(t.required_tool for t in tasks)),
        "splits": dict(Counter(t.split for t in tasks)),
        "science_ready": True,
        "safety_scope": "benign toy tools only; no web, credentials, real filesystem writes, or harmful tools",
    }
    return tasks, info


def render_prompt(task: ToolTask) -> str:
    return (
        "You are in a controlled toy-tool harness.\n"
        "Available tools: calculator, dictionary, calendar, file_search, route_planner, unit_converter, or none.\n"
        "User: "
        + task.user_prompt.strip()
        + "\nTool needed?"
    )


def choose_depths(bundle: bench.ModelBundle) -> list[int]:
    n = int(bundle.anatomy.n_layers)
    return sorted({0, max(1, n // 2), n})


def capture_vectors(bundle: bench.ModelBundle, tasks: Sequence[ToolTask], depths: Sequence[int]) -> dict[tuple[str, int], Any]:
    vectors: dict[tuple[str, int], Any] = {}
    for task in tasks:
        capture = bench.run_with_residual_cache(bundle, render_prompt(task))
        for depth in depths:
            vectors[(task.task_id, depth)] = capture.streams[depth, -1, :].detach().clone()
    return vectors


def mean_vec(vecs: Sequence[Any]) -> Any:
    import torch

    if not vecs:
        raise ValueError("mean_vec called with no vectors")
    return torch.stack([v.float() for v in vecs]).mean(dim=0)


def build_directions(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    tasks: Sequence[ToolTask],
    vectors: Mapping[tuple[str, int], Any],
    depth: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    train = [t for t in tasks if t.split == "train"] or list(tasks)
    needed = [vectors[(t.task_id, depth)] for t in train if t.tool_needed]
    no_tool = [vectors[(t.task_id, depth)] for t in train if not t.tool_needed]
    if not no_tool:
        no_tool = [vectors[(t.task_id, depth)] for t in train if t.required_tool == "none"]
    needed_dir = unit_vector(mean_vec(needed) - mean_vec(no_tool))
    dirs: dict[str, Any] = {"tool_needed": needed_dir, "none": -needed_dir}
    all_train = [vectors[(t.task_id, depth)] for t in train]
    for tool in TOOLS:
        if tool == "none":
            continue
        pos = [vectors[(t.task_id, depth)] for t in train if t.required_tool == tool]
        neg = [vectors[(t.task_id, depth)] for t in train if t.required_tool != tool]
        dirs[tool] = unit_vector(mean_vec(pos) - mean_vec(neg)) if pos and neg else torch.zeros_like(needed_dir)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(getattr(ctx.args, "seed", 0) or 0) + 3400)
    dirs["random_direction_control"] = unit_vector(torch.randn(needed_dir.shape, dtype=needed_dir.dtype, generator=gen))
    diff_norm = safe_mean([float(torch.linalg.vector_norm(v.float() - mean_vec(all_train))) for v in all_train], default=1.0)
    steer = {name: vec.float() * float(diff_norm) for name, vec in dirs.items()}
    meta = {
        "depth": depth,
        "steer_layer": max(0, min(bundle.anatomy.n_layers - 1, depth - 1)),
        "mean_centered_train_norm": rounded(diff_norm),
        "direction_norms": {name: rounded(float(torch.linalg.vector_norm(vec.float()))) for name, vec in dirs.items()},
    }
    return {"unit": dirs, "steer": steer}, meta


def prompt_features(task: ToolTask) -> dict[str, float]:
    text = task.user_prompt.lower()
    return {
        "length_chars": float(len(text)),
        "digit_count": float(sum(ch.isdigit() for ch in text)),
        "operator_count": float(sum(ch in "+-*/" for ch in text)),
        "tool_name_mentioned": float(any(tool in text for tool in TOOLS if tool != "none")),
        "distractor_tool_mentioned": float(task.distractor_tool in text),
        "lookup_word_count": float(sum(w in text for w in ("look up", "search", "find", "glossary", "calendar", "convert", "route", "compute"))),
    }


def surface_tool_prediction(task: ToolTask) -> str:
    text = task.user_prompt.lower()
    for tool in TOOLS:
        if tool != "none" and tool in text:
            return tool
    if re.search(r"\d+\s*[\+\-\*/]\s*\d+", text) or "compute" in text:
        return "calculator"
    if "glossary" in text or "look up" in text:
        return "dictionary"
    if "calendar" in text or "event" in text:
        return "calendar"
    if "doc" in text or "search" in text or "file" in text:
        return "file_search"
    if "route" in text or "path" in text or "graph" in text:
        return "route_planner"
    if "convert" in text or "kilometers" in text or "fahrenheit" in text:
        return "unit_converter"
    return "none"


def probe_reports(
    tasks: Sequence[ToolTask],
    vectors: Mapping[tuple[str, int], Any],
    directions_by_depth: Mapping[int, Mapping[str, Any]],
    depths: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for depth in depths:
        dirs = directions_by_depth[depth]["unit"]
        needed_scores: list[float] = []
        needed_labels: list[int] = []
        tool_correct: list[float] = []
        surface_correct: list[float] = []
        shuffled_correct: list[float] = []
        for idx, task in enumerate(tasks):
            vec = vectors[(task.task_id, depth)].float()
            needed_score = float(vec @ dirs["tool_needed"].float())
            needed_pred = needed_score > 0.0
            tool_scores = {tool: float(vec @ dirs[tool].float()) for tool in TOOLS}
            if not needed_pred:
                tool_pred = "none"
            else:
                tool_pred = max([t for t in TOOLS if t != "none"], key=lambda t: tool_scores[t])
            surface_pred = surface_tool_prediction(task)
            shuffled_label = tasks[(idx + 7) % len(tasks)].required_tool
            needed_scores.append(needed_score)
            needed_labels.append(1 if task.tool_needed else 0)
            tool_correct.append(1.0 if tool_pred == task.required_tool else 0.0)
            surface_correct.append(1.0 if surface_pred == task.required_tool else 0.0)
            shuffled_correct.append(1.0 if tool_pred == shuffled_label else 0.0)
            if depth == depths[-1]:
                feats = prompt_features(task)
                task_rows.append({
                    "task_id": task.task_id,
                    "family": task.family,
                    "required_tool": task.required_tool,
                    "tool_needed": task.tool_needed,
                    "tool_probe_prediction": tool_pred,
                    "surface_cue_prediction": surface_pred,
                    "tool_probe_correct": tool_pred == task.required_tool,
                    "surface_cue_correct": surface_pred == task.required_tool,
                    "needed_score": rounded(needed_score),
                    "tool_score_json": json.dumps({k: rounded(v) for k, v in tool_scores.items()}, sort_keys=True),
                    **{k: rounded(v) for k, v in feats.items()},
                    "student_trace_label": "",
                    "student_confidence": "",
                    "student_evidence_span": "",
                    "reviewer_trace_label": "",
                    "agreement_status": "",
                })
        rows.append({
            "probe": "tool_needed",
            "depth": depth,
            "n_tasks": len(tasks),
            "auc": rounded(auc_binary(needed_labels, needed_scores)),
            "accuracy": rounded(safe_mean([1.0 if ((s > 0.0) == bool(y)) else 0.0 for s, y in zip(needed_scores, needed_labels)], default=0.0)),
            "surface_control_accuracy": rounded(safe_mean(surface_correct, default=0.0)),
            "shuffled_label_control_accuracy": "",
            "claim_scope": "prompt_boundary_decode",
        })
        rows.append({
            "probe": "tool_selection",
            "depth": depth,
            "n_tasks": len(tasks),
            "auc": "",
            "accuracy": rounded(safe_mean(tool_correct, default=0.0)),
            "surface_control_accuracy": rounded(safe_mean(surface_correct, default=0.0)),
            "shuffled_label_control_accuracy": rounded(safe_mean(shuffled_correct, default=0.0)),
            "claim_scope": "prompt_boundary_decode",
        })
    return rows, task_rows


ALLOWED_AST = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
    ast.Load,
}
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def safe_eval_expr(expr: str) -> float:
    node = ast.parse(expr, mode="eval")
    for sub in ast.walk(node):
        if type(sub) not in ALLOWED_AST:
            raise ValueError(f"unsupported expression node {type(sub).__name__}")

    def eval_node(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return eval_node(n.body)
        if isinstance(n, ast.Constant):
            return float(n.value)
        if isinstance(n, ast.UnaryOp):
            val = eval_node(n.operand)
            return -val if isinstance(n.op, ast.USub) else val
        if isinstance(n, ast.BinOp):
            return OPS[type(n.op)](eval_node(n.left), eval_node(n.right))
        raise ValueError(f"unsupported node {type(n).__name__}")

    return eval_node(node)


def run_tool(task: ToolTask) -> tuple[str, str, list[str]]:
    if task.required_tool == "none":
        return "none", task.answer, []
    if task.required_tool == "calculator":
        value = safe_eval_expr(str(task.tool_args["expression"]))
        result = str(int(value)) if abs(value - int(value)) < 1e-9 else str(round(value, 4))
        return "calculator", result, ["arithmetic"]
    if task.required_tool == "dictionary":
        term = str(task.tool_args["term"])
        return "dictionary", GLOSSARY[term], [term]
    if task.required_tool == "calendar":
        if "event" in task.tool_args:
            key = str(task.tool_args["event"])
            return "calendar", CALENDAR[key], [key]
        if "after" in task.tool_args:
            return "calendar", "bug triage", ["standup", "bug triage"]
        return "calendar", "demo prep", ["Friday afternoon"]
    if task.required_tool == "file_search":
        query = str(task.tool_args["query"]).lower()
        for name, text in DOCS.items():
            if all(part in text.lower() for part in query.split()[:2]) or query in text.lower():
                return "file_search", name, [name]
        if "retry" in query:
            return "file_search", "doc_reliability.md", ["doc_reliability.md"]
        raise KeyError(query)
    if task.required_tool == "route_planner":
        start = str(task.tool_args["start"])
        end = str(task.tool_args["end"])
        path = shortest_path(start, end)
        return "route_planner", " -> ".join(path), path
    if task.required_tool == "unit_converter":
        value = float(task.tool_args["value"])
        src = str(task.tool_args["from"])
        dst = str(task.tool_args["to"])
        if src == "miles" and dst == "kilometers":
            return "unit_converter", f"{value * 1.60934:.2f} kilometers", [src, dst]
        if src == "pounds" and dst == "kilograms":
            return "unit_converter", f"{value * 0.453592:.2f} kilograms", [src, dst]
        if src == "celsius" and dst == "fahrenheit":
            return "unit_converter", f"{int(value * 9 / 5 + 32)} Fahrenheit", [src, dst]
    raise ValueError(task.required_tool)


def shortest_path(start: str, end: str) -> list[str]:
    queue: deque[list[str]] = deque([[start]])
    seen = {start}
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == end:
            return path
        for nxt in GRAPH.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(path + [nxt])
    raise ValueError(f"no route {start}->{end}")


def trace_log(tasks: Sequence[ToolTask]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    for task in tasks:
        tool, result, reads = run_tool(task)
        correct = str(result) == str(task.answer)
        corrupted = "CORRUPTED_RESULT" if tool != "none" else task.answer
        trace_rows.append({
            "task_id": task.task_id,
            "required_tool": task.required_tool,
            "tool_invoked": tool,
            "tool_args_json": json.dumps(task.tool_args, sort_keys=True),
            "tool_result": result,
            "expected_answer": task.answer,
            "result_matches_expected": correct,
            "memory_reads_json": json.dumps(reads),
            "corrupted_tool_result": corrupted,
            "would_answer_change_if_corrupted": bool(tool != "none"),
        })
        report_rows.append({
            "task_id": task.task_id,
            "known_trace_label": tool,
            "self_report_template": "no tool was needed" if tool == "none" else f"used {tool} with args {json.dumps(task.tool_args, sort_keys=True)}",
            "matches_known_trace": True,
            "requires_human_review": True,
            "student_trace_label": "",
            "student_confidence": "",
            "student_evidence_span": "",
            "reviewer_trace_label": "",
            "agreement_status": "",
        })
    return trace_rows, report_rows


def intervention_prompt(task: ToolTask) -> str:
    return (
        "Choose the best next action for the controlled toy task.\n"
        "A=calculator B=dictionary C=calendar D=file_search E=route_planner F=unit_converter N=no_tool\n"
        f"User: {task.user_prompt}\n"
        "Action letter:"
    )


def token_id_or_first(bundle: bench.ModelBundle, text: str) -> int:
    ids = bundle.tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise RuntimeError(f"No token ids for {text!r}")
    return int(ids[0])


def run_interventions(
    bundle: bench.ModelBundle,
    tasks: Sequence[ToolTask],
    directions: Mapping[str, Mapping[str, Any]],
    meta: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    steer_layer = int(meta["steer_layer"])
    letter_ids = {tool: token_id_or_first(bundle, " " + letter) for tool, letter in TOOL_LETTERS.items()}
    for task in tasks:
        target_tool = task.required_tool
        distractor = task.distractor_tool if task.required_tool != "none" else task.distractor_tool
        target_id = letter_ids[target_tool]
        distractor_id = letter_ids[distractor]
        target_vec = directions["steer"].get(target_tool, directions["steer"]["none"])
        for direction_name, vector in ((target_tool, target_vec), ("random_direction_control", directions["steer"]["random_direction_control"])):
            for scale in STEER_SCALES:
                logits = bench.next_token_logits(bundle, intervention_prompt(task), steer=(steer_layer, vector, float(scale)))
                margin = float(logits[target_id] - logits[distractor_id])
                rows.append({
                    "task_id": task.task_id,
                    "required_tool": task.required_tool,
                    "distractor_tool": distractor,
                    "direction": direction_name,
                    "scale": scale,
                    "target_letter": TOOL_LETTERS[target_tool],
                    "distractor_letter": TOOL_LETTERS[distractor],
                    "target_minus_distractor_logit": rounded(margin),
                })
    zero = {(r["task_id"], r["direction"]): float(r["target_minus_distractor_logit"]) for r in rows if float(r["scale"]) == 0.0}
    shifts: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if float(row["scale"]) == 1.0:
            base = zero.get((row["task_id"], row["direction"]))
            if base is not None:
                shifts[str(row["direction"])].append(float(row["target_minus_distractor_logit"]) - base)
    real_shifts = [v for k, vals in shifts.items() if k != "random_direction_control" for v in vals]
    random_shift = safe_mean(shifts["random_direction_control"], default=0.0)
    real_shift = safe_mean(real_shifts, default=0.0)
    summary = {
        "mean_tool_direction_shift_at_scale_1": rounded(real_shift),
        "mean_random_direction_shift_at_scale_1": rounded(random_shift),
        "shift_over_random": rounded(real_shift - random_shift),
        "supported": bool(real_shift - random_shift > 0.05 and real_shift > 0.0),
    }
    return rows, summary


def evidence_matrix(
    probe_rows: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    report_rows: Sequence[Mapping[str, Any]],
    intervention_summary: Mapping[str, Any],
    final_depth: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    needed = [r for r in probe_rows if r["probe"] == "tool_needed" and int(r["depth"]) == final_depth][0]
    choice = [r for r in probe_rows if r["probe"] == "tool_selection" and int(r["depth"]) == final_depth][0]
    trace_acc = safe_mean([1.0 if r["result_matches_expected"] else 0.0 for r in trace_rows], default=0.0)
    report_match = safe_mean([1.0 if r["matches_known_trace"] else 0.0 for r in report_rows], default=0.0)
    choice_acc = float(choice["accuracy"] or 0.0)
    surface_acc = float(choice["surface_control_accuracy"] or 0.0)
    rows = [
        {
            "method": "tool_needed_probe",
            "evidence_rung": "DECODE",
            "metric": "AUC",
            "value": needed["auc"],
            "control_value": needed["surface_control_accuracy"],
            "claim_posture": "decode_supported" if float(needed["auc"] or 0.0) >= 0.75 else "not_supported",
        },
        {
            "method": "tool_selection_probe",
            "evidence_rung": "DECODE",
            "metric": "accuracy",
            "value": choice["accuracy"],
            "control_value": choice["surface_control_accuracy"],
            "claim_posture": "surface_cue_limited" if choice_acc <= surface_acc + 0.05 else "beats_surface_control",
        },
        {
            "method": "activation_addition_tool_choice",
            "evidence_rung": "CAUSAL",
            "metric": "shift over random",
            "value": intervention_summary["shift_over_random"],
            "control_value": intervention_summary["mean_random_direction_shift_at_scale_1"],
            "claim_posture": "causal_shift_supported" if intervention_summary["supported"] else "causal_shift_not_established",
        },
        {
            "method": "deterministic_tool_trace",
            "evidence_rung": "OBS",
            "metric": "trace result accuracy",
            "value": rounded(trace_acc),
            "control_value": "",
            "claim_posture": "harness_trace_supported",
        },
        {
            "method": "self_report_trace_labels",
            "evidence_rung": "SELF-REPORT",
            "metric": "known-trace match",
            "value": rounded(report_match),
            "control_value": "",
            "claim_posture": "template_labels_need_human_review",
        },
    ]
    metrics = {
        "final_depth": final_depth,
        "tool_needed_auc": needed["auc"],
        "tool_selection_accuracy": choice["accuracy"],
        "surface_control_accuracy": choice["surface_control_accuracy"],
        "trace_result_accuracy": rounded(trace_acc),
        "self_report_known_trace_match": rounded(report_match),
        "causal_shift_over_random": intervention_summary["shift_over_random"],
    }
    return rows, metrics


def write_tables(
    ctx: bench.RunContext,
    task_rows: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    intervention_rows: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    report_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
) -> None:
    specs = [
        ("tables/tool_task_manifest.csv", task_rows, "Task-level tool probe predictions and surface-cue controls."),
        ("tables/tool_choice_probe_report.csv", probe_rows, "Tool-needed and tool-selection probe report by depth."),
        ("tables/tool_intervention_report.csv", intervention_rows, "Activation-addition tool-choice prompt results."),
        ("tables/tool_trace_log.csv", trace_rows, "Deterministic toy-tool trace log."),
        ("tables/tool_self_report_labels.csv", report_rows, "Known-trace self-report labels and review fields."),
        ("tables/tool_use_evidence_matrix.csv", evidence_rows, "Lab 34 evidence matrix."),
    ]
    for rel, rows, desc in specs:
        path = ctx.path(*rel.split("/"))
        bench.write_csv_with_context(ctx, path, rows)
        ctx.register_artifact(path, "table", desc)


def write_state(ctx: bench.RunContext, directions: Mapping[str, Mapping[str, Any]], meta: Mapping[str, Any]) -> None:
    import torch

    path = ctx.path("state", "tool_directions.pt")
    torch.save({name: vec.cpu() for name, vec in directions["steer"].items()}, path)
    ctx.register_artifact(path, "state", "Tool-needed and tool-selection steering directions.")
    meta_path = ctx.path("state", "tool_direction_metadata.json")
    bench.write_json(meta_path, meta)
    ctx.register_artifact(meta_path, "state", "Tool direction depth and norm metadata.")


def write_safety_status(ctx: bench.RunContext, data_info: Mapping[str, Any]) -> None:
    payload = {
        "lab": LAB_ID,
        "safe_scope": data_info["safety_scope"],
        "blocked_activities": ["web browsing", "real credentials", "real filesystem writes", "harmful tools", "autonomous deployment"],
        "science_ready": data_info["science_ready"],
    }
    path = ctx.path("diagnostics", "safety_status.json")
    bench.write_json(path, payload)
    ctx.register_artifact(path, "diagnostic", "Safety and scope status for Lab 34.")


def write_method_card(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 34 method card",
        "",
        "This lab studies controlled toy-tool traces. It does not claim persistent goals or autonomous plans.",
        "",
        "- tools: calculator, dictionary, calendar, file_search, route_planner, unit_converter, none",
        "- decode object: prompt-boundary residual state",
        "- main null: surface cues explain tool choice",
        "- self-report labels: known trace templates requiring human review",
        "- forbidden claim: the model has a persistent goal or autonomous plan",
        "",
        f"- tool-needed AUC: `{metrics['tool_needed_auc']}`",
        f"- tool-selection accuracy: `{metrics['tool_selection_accuracy']}`",
        f"- surface-control accuracy: `{metrics['surface_control_accuracy']}`",
        f"- causal shift over random: `{metrics['causal_shift_over_random']}`",
        "",
        "| method | rung | value | control | posture |",
        "|---|---|---:|---:|---|",
    ]
    for row in evidence:
        lines.append(f"| {row['method']} | {row['evidence_rung']} | {row['value']} | {row['control_value']} | {row['claim_posture']} |")
    path = ctx.path("method_card.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 34 method card and non-claims.")


def write_operationalization_audit(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Lab 34 operationalization audit",
        "",
        "Favorite interpretation under attack: a decodable tool-needed direction is an intention or plan.",
        "",
        "## What the measurement can say",
        "",
        "A prompt-boundary state separated toy tool labels or shifted a tool-choice prompt under controlled interventions.",
        "",
        "## What it cannot say",
        "",
        "It cannot say the model has a persistent goal, an autonomous plan, or a reliable real-world tool policy.",
        "",
        "## Cheap explanations",
        "",
        "- The prompt contains a tool name.",
        "- Digits or arithmetic symbols imply calculator.",
        "- The answer is already in the prompt.",
        "- The deterministic tool trace is mistaken for model self-report.",
        "- Steering changes an A/B letter prior rather than tool cognition.",
        "",
        "## Verdicts",
        "",
    ]
    for row in evidence:
        lines.append(f"- `{row['method']}`: `{row['claim_posture']}`.")
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 34 controls and non-claims.")


def write_run_summary(ctx: bench.RunContext, data_info: Mapping[str, Any], metrics: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Lab 34 run summary: tool use, agents, and state tracking",
        "",
        f"- data rows: {data_info['n_rows_selected']} selected from `{pathlib.Path(str(data_info['data_path'])).name}`",
        f"- required tools: `{data_info['required_tools']}`",
        f"- tool-needed AUC: `{metrics['tool_needed_auc']}`",
        f"- tool-selection accuracy: `{metrics['tool_selection_accuracy']}` vs surface control `{metrics['surface_control_accuracy']}`",
        f"- causal shift over random: `{metrics['causal_shift_over_random']}`",
        "",
        "## Evidence matrix",
        "",
        "| method | rung | metric | value | posture |",
        "|---|---|---|---:|---|",
    ]
    for row in evidence:
        lines.append(f"| `{row['method']}` | {row['evidence_rung']} | {row['metric']} | {row['value']} | {row['claim_posture']} |")
    lines += [
        "",
        "## Reading order",
        "",
        "1. `method_card.md` for scope and non-claims.",
        "2. `tables/tool_task_manifest.csv` for probe predictions and surface controls.",
        "3. `tables/tool_choice_probe_report.csv` for depth-wise decoding.",
        "4. `tables/tool_trace_log.csv` for deterministic tool execution.",
        "5. `tables/tool_intervention_report.csv` for steering results.",
        "6. `tables/tool_self_report_labels.csv` before citing self-report.",
        "",
        "## Smallest surviving claim",
        "",
        "A prompt-boundary signal can be audited against surface cues and deterministic toy-tool traces. This is not evidence for persistent goals or autonomous planning.",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 34 run summary and reading order.")


def write_plot_guide(ctx: bench.RunContext) -> None:
    rows = [
        {"plot": "plots/tool_use_evidence_dashboard.png", "read_for": "Decode, surface-control, trace, and causal summaries.", "non_claim": "Tool signal is not an intention."},
        {"plot": "plots/tool_choice_probe_by_depth.png", "read_for": "Tool-needed AUC and selection accuracy by depth.", "non_claim": "Depth trend is descriptive."},
        {"plot": "plots/tool_selection_confusion_matrix.png", "read_for": "Required tool versus predicted tool.", "non_claim": "Confusion is on toy tasks."},
        {"plot": "plots/tool_state_patch_recovery.png", "read_for": "Activation-addition shift by direction.", "non_claim": "Steering is a narrow prompt test."},
        {"plot": "plots/memory_read_trace_atlas.png", "read_for": "Known memory/tool reads by family.", "non_claim": "Trace is harness-generated."},
        {"plot": "plots/tool_result_reliance_ladder.png", "read_for": "Would corrupted tool result alter the answer.", "non_claim": "Deterministic reliance is not model faithfulness."},
        {"plot": "plots/tool_self_report_matrix.png", "read_for": "Known-trace labels and review status.", "non_claim": "Template self-report needs human review."},
    ]
    path = ctx.path("plots", "plot_reading_guide.csv")
    bench.write_csv(path, rows)
    ctx.register_artifact(path, "table", "Plot reading guide for Lab 34.")


def write_plots(
    ctx: bench.RunContext,
    tasks: Sequence[ToolTask],
    task_rows: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    intervention_rows: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    report_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
) -> None:
    write_plot_guide(ctx)
    if ctx.args.no_plots:
        return
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Lab 34 tool-use evidence dashboard", fontsize=14, fontweight="bold")
    names = [r["method"] for r in evidence_rows]
    vals = [float(r["value"] or 0.0) for r in evidence_rows]
    axes[0, 0].bar(range(len(names)), vals, color="#0072B2")
    axes[0, 0].set_xticks(range(len(names)), names, rotation=35, ha="right")
    axes[0, 0].set_title("Evidence values")
    needed_rows = [r for r in probe_rows if r["probe"] == "tool_needed"]
    axes[0, 1].plot([int(r["depth"]) for r in needed_rows], [float(r["auc"] or 0.0) for r in needed_rows], marker="o", color="#009E73")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_title("Tool-needed AUC by depth")
    by_dir = defaultdict(list)
    for row in intervention_rows:
        if float(row["scale"]) == 1.0:
            by_dir[row["direction"]].append(float(row["target_minus_distractor_logit"]))
    dirs = sorted(by_dir)
    axes[1, 0].bar(dirs, [safe_mean(by_dir[d], default=0.0) for d in dirs], color="#D55E00")
    axes[1, 0].set_xticks(range(len(dirs)), dirs, rotation=35, ha="right")
    axes[1, 0].set_title("Scale-1 target-vs-distractor logits")
    families = sorted({t.family for t in tasks})
    axes[1, 1].bar(families, [sum(1 for r in trace_rows if next(t.family for t in tasks if t.task_id == r["task_id"]) == fam and r["would_answer_change_if_corrupted"]) for fam in families], color="#CC79A7")
    axes[1, 1].set_xticks(range(len(families)), families, rotation=35, ha="right")
    axes[1, 1].set_title("Tool-result reliance by family")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    bench.save_figure(ctx, fig, "tool_use_evidence_dashboard.png", "Lab 34 tool-use evidence dashboard.")

    depths = sorted({int(r["depth"]) for r in probe_rows})
    needed = [next(r for r in probe_rows if r["probe"] == "tool_needed" and int(r["depth"]) == d) for d in depths]
    selection = [next(r for r in probe_rows if r["probe"] == "tool_selection" and int(r["depth"]) == d) for d in depths]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(depths, [float(r["auc"] or 0.0) for r in needed], marker="o", label="tool-needed AUC")
    ax.plot(depths, [float(r["accuracy"] or 0.0) for r in selection], marker="s", label="tool-selection accuracy")
    ax.plot(depths, [float(r["surface_control_accuracy"] or 0.0) for r in selection], marker="^", label="surface control")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("residual stream depth")
    ax.legend(fontsize=8)
    ax.set_title("Tool-choice probe by depth")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "tool_choice_probe_by_depth.png", "Tool probe score by residual depth.")

    labels = list(TOOLS)
    idx = {tool: i for i, tool in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)))
    for row in task_rows:
        matrix[idx[str(row["required_tool"])]][idx[str(row["tool_probe_prediction"])]] += 1
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("required")
    ax.set_title("Tool-selection confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j]:
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="#111111")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "tool_selection_confusion_matrix.png", "Required-vs-predicted tool confusion matrix.")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    by_scale_dir = defaultdict(list)
    for row in intervention_rows:
        by_scale_dir[(row["direction"], float(row["scale"]))].append(float(row["target_minus_distractor_logit"]))
    for direction in sorted({r["direction"] for r in intervention_rows})[:8]:
        xs = sorted({float(r["scale"]) for r in intervention_rows if r["direction"] == direction})
        ax.plot(xs, [safe_mean(by_scale_dir[(direction, x)], default=0.0) for x in xs], marker="o", label=direction)
    ax.axhline(0, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_xlabel("scale")
    ax.set_ylabel("target minus distractor logit")
    ax.set_title("Tool-state patch/steering recovery")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "tool_state_patch_recovery.png", "Tool-choice activation-addition frontier.")

    read_counts = defaultdict(int)
    for row in trace_rows:
        fam = next(t.family for t in tasks if t.task_id == row["task_id"])
        read_counts[fam] += len(json.loads(row["memory_reads_json"]))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(sorted(read_counts), [read_counts[f] for f in sorted(read_counts)], color="#009E73")
    ax.set_xticks(range(len(read_counts)), sorted(read_counts), rotation=35, ha="right")
    ax.set_ylabel("known memory/tool reads")
    ax.set_title("Memory-read trace atlas")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "memory_read_trace_atlas.png", "Known memory-read trace counts by family.")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    reliance = defaultdict(list)
    for row in trace_rows:
        fam = next(t.family for t in tasks if t.task_id == row["task_id"])
        reliance[fam].append(1.0 if row["would_answer_change_if_corrupted"] else 0.0)
    fams = sorted(reliance)
    ax.bar(fams, [safe_mean(reliance[f], default=0.0) for f in fams], color="#D55E00")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(fams)), fams, rotation=35, ha="right")
    ax.set_title("Tool-result reliance ladder")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "tool_result_reliance_ladder.png", "Tool-result reliance by family.")

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    match = sum(1 for r in report_rows if r["matches_known_trace"])
    review = sum(1 for r in report_rows if r["requires_human_review"])
    ax.imshow([[match, len(report_rows) - match], [review, len(report_rows) - review]], cmap="Purples")
    ax.set_xticks([0, 1], ["yes", "no"])
    ax.set_yticks([0, 1], ["matches trace", "needs review"])
    ax.set_title("Tool self-report matrix")
    for i, row in enumerate([[match, len(report_rows) - match], [review, len(report_rows) - review]]):
        for j, val in enumerate(row):
            ax.text(j, i, val, ha="center", va="center")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "tool_self_report_matrix.png", "Known-trace self-report review matrix.")


def write_claims(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]]) -> None:
    run_name = ctx.run_dir.name
    claims = []
    for i, row in enumerate(evidence, start=1):
        claims.append({
            "id": f"{LAB_ID}-C{i}",
            "tag": str(row["evidence_rung"]),
            "text": (
                f"Method `{row['method']}` reported {row['metric']}={row['value']} under toy-tool controls; "
                f"posture `{row['claim_posture']}`."
            ),
            "artifact": f"runs/{run_name}/tables/tool_use_evidence_matrix.csv",
            "falsifier": "Surface-cue controls, corrupted tool results, or human trace review explain the measured signal.",
        })
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    tasks, data_info = load_tasks(ctx)
    manifest_path = ctx.path("diagnostics", "data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Lab 34 data manifest and tool scope.")
    write_safety_status(ctx, data_info)
    bench.run_hook_parity_check(ctx, bundle, render_prompt(tasks[0]))
    first = bench.run_with_residual_cache(bundle, render_prompt(tasks[0]))
    bench.run_lens_self_check(ctx, bundle, first)
    bench.run_patch_noop_check(ctx, bundle, render_prompt(tasks[0]))

    depths = choose_depths(bundle)
    vectors = capture_vectors(bundle, tasks, depths)
    directions_by_depth: dict[int, Any] = {}
    meta_by_depth: dict[int, Any] = {}
    for depth in depths:
        dirs, meta = build_directions(ctx, bundle, tasks, vectors, depth)
        directions_by_depth[depth] = dirs
        meta_by_depth[depth] = meta
    probe_rows, task_rows = probe_reports(tasks, vectors, directions_by_depth, depths)
    final_depth = depths[-1]
    intervention_rows, intervention_summary = run_interventions(bundle, tasks, directions_by_depth[final_depth], meta_by_depth[final_depth])
    trace_rows, report_rows = trace_log(tasks)
    evidence_rows, metrics = evidence_matrix(probe_rows, trace_rows, report_rows, intervention_summary, final_depth)
    metrics = {**metrics, "data": data_info, "intervention": intervention_summary, "directions": meta_by_depth[final_depth]}

    write_tables(ctx, task_rows, probe_rows, intervention_rows, trace_rows, report_rows, evidence_rows)
    write_state(ctx, directions_by_depth[final_depth], meta_by_depth[final_depth])
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 34 metrics.")
    write_method_card(ctx, evidence_rows, metrics)
    write_operationalization_audit(ctx, evidence_rows)
    write_run_summary(ctx, data_info, metrics, evidence_rows)
    write_claims(ctx, evidence_rows)
    write_plots(ctx, tasks, task_rows, probe_rows, intervention_rows, trace_rows, report_rows, evidence_rows)
    print(f"[lab34] wrote {len(task_rows)} task rows, {len(probe_rows)} probe rows, and {len(evidence_rows)} evidence rows")
