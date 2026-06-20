"""Lab 34: Tool use, agents, and state tracking.

This lab studies a controlled toy-tool harness. The scientific object is narrow:
prompt-boundary residual states, deterministic local tool traces, and a
constrained action-letter intervention. The lab deliberately avoids real tools,
web browsing, credentials, real filesystem access, harmful tasks, and autonomous
agent claims.

Evidence level: OBS + DECODE + CAUSAL + SELF-REPORT, scoped to benign toy tools.
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
from collections.abc import Mapping, Sequence
from typing import Any

import interp_bench as bench

LAB_ID = "L34"
LAB_NAME = "lab34_tool_use_state"
DATA_FILE = "tool_use_tasks.jsonl"
PROMPT_SET_CAPS = {"small": 28, "medium": 42, "full": 0}
SCIENCE_READY_MIN_ROWS = 70
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
LETTER_TO_TOOL = {v: k for k, v in TOOL_LETTERS.items()}
STEER_SCALES = (-1.0, 0.0, 0.5, 1.0, 1.5)
N_RANDOM_INTERVENTION_CONTROLS = 5
CLAIMABLE_SCALE = 1.0
DECODE_AUC_BAR = 0.60
DECODE_GAP_BAR = 0.05
CAUSAL_GAP_BAR = 0.05
RESIDUAL_VECTOR_SCALE_FLOOR = 1e-8
REVIEW_FIELDS = (
    "student_trace_label",
    "student_confidence",
    "student_evidence_span",
    "reviewer_trace_label",
    "agreement_status",
)

GLOSSARY = {
    "latency": "delay before a response begins",
    "mutex": "a lock that allows one holder at a time",
    "photosynthesis": "plants use light to make sugar from water and carbon dioxide",
    "backoff": "waiting longer between retries after a failure",
    "vector": "an ordered list of numbers treated as one object",
    "cache": "stored results reused to avoid repeated work",
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


@dataclasses.dataclass(frozen=True)
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
    notes: str = ""
    surface_cues: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ToolExecution:
    tool_invoked: str
    result: str
    memory_reads: tuple[str, ...]
    argument_valid: bool
    error: str = ""


@dataclasses.dataclass(frozen=True)
class DirectionModel:
    depth: int
    steer_layer: int
    unit: dict[str, Any]
    steer: dict[str, Any]
    needed_threshold: float
    train_center_norm: float
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


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


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        f = float(value)
    except Exception:
        return default
    return f if math.isfinite(f) else default


def safe_mean(values: Sequence[Any], default: float = float("nan")) -> float:
    vals: list[float] = []
    for value in values:
        f = as_float(value)
        if math.isfinite(f):
            vals.append(f)
    return float(statistics.fmean(vals)) if vals else default


def safe_stdev(values: Sequence[Any], default: float = float("nan")) -> float:
    vals = [as_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if len(vals) < 2:
        return default
    return float(statistics.stdev(vals))


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
    if float(norm) <= RESIDUAL_VECTOR_SCALE_FLOOR:
        return torch.zeros_like(vec.float())
    return vec.float() / norm


def mean_vec(vecs: Sequence[Any]) -> Any:
    import torch

    if not vecs:
        raise ValueError("mean_vec called with no vectors")
    return torch.stack([v.float() for v in vecs]).mean(dim=0)


def best_binary_threshold(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = [(int(y), float(s)) for y, s in zip(labels, scores) if math.isfinite(float(s))]
    if not pairs:
        return 0.0
    unique = sorted({s for _, s in pairs})
    candidates = [unique[0] - 1e-6, unique[-1] + 1e-6]
    candidates.extend((a + b) / 2.0 for a, b in zip(unique, unique[1:]))
    best = max(candidates, key=lambda thr: sum(1 for y, s in pairs if (s >= thr) == bool(y)))
    return float(best)


def task_sort_key(task: ToolTask) -> tuple[Any, ...]:
    return (task.required_tool, task.split, stable_int(task.task_id))


def write_jsonl(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), sort_keys=True, default=bench.json_default) + "\n")


# ---------------------------------------------------------------------------
# Data loading, prompt rendering, and validation
# ---------------------------------------------------------------------------


def data_path(args: Any) -> pathlib.Path:
    prompt_set = str(getattr(args, "prompt_set", "") or "")
    candidate = pathlib.Path(prompt_set)
    if prompt_set not in PROMPT_SET_CAPS and candidate.suffix.lower() in {".jsonl", ".json"}:
        return candidate if candidate.is_absolute() else (bench.COURSE_ROOT / candidate).resolve()
    return bench.COURSE_ROOT / "data" / DATA_FILE


def builtin_smoke_rows() -> list[dict[str, Any]]:
    """Tiny fallback used only when the frozen JSONL is absent in Tier A."""
    base = [
        ("calc_smoke_train", "calculator", "What is 6 * 7?", True, {"expression": "6 * 7"}, "42", "unit_converter", "train"),
        ("dict_smoke_train", "dictionary", "Look up latency in the toy glossary.", True, {"term": "latency"}, "delay before a response begins", "file_search", "train"),
        ("cal_smoke_train", "calendar", "When is standup on the toy calendar?", True, {"event": "standup"}, "Monday 09:00", "dictionary", "train"),
        ("file_smoke_train", "file_search", "Find the synthetic doc about retry budget.", True, {"query": "retry budget"}, "doc_reliability.md", "dictionary", "train"),
        ("route_smoke_train", "route_planner", "Plan a route from A to D.", True, {"start": "A", "end": "D"}, "A -> B -> D", "calendar", "train"),
        ("unit_smoke_train", "unit_converter", "Convert 2 hours to minutes.", True, {"from": "hours", "to": "minutes", "value": 2}, "120 minutes", "calculator", "train"),
        ("none_smoke_train", "no_tool", "The word calculator appears here. Answer banana.", False, {}, "banana", "calculator", "train"),
        ("calc_smoke_eval", "calculator", "Compute 9 + 5.", True, {"expression": "9 + 5"}, "14", "unit_converter", "eval"),
        ("dict_smoke_eval", "dictionary", "Define mutex using the toy dictionary.", True, {"term": "mutex"}, "a lock that allows one holder at a time", "file_search", "eval"),
        ("cal_smoke_eval", "calendar", "Find demo prep on the toy calendar.", True, {"event": "demo prep"}, "Friday afternoon", "dictionary", "eval"),
        ("file_smoke_eval", "file_search", "Which file mentions cache invalidation?", True, {"query": "cache invalidation"}, "doc_cache.md", "dictionary", "eval"),
        ("route_smoke_eval", "route_planner", "Route C to F in the toy graph.", True, {"start": "C", "end": "F"}, "C -> E -> F", "calendar", "eval"),
        ("unit_smoke_eval", "unit_converter", "Convert 3 miles to kilometers.", True, {"from": "miles", "to": "kilometers", "value": 3}, "4.83 kilometers", "calculator", "eval"),
        ("none_smoke_eval", "no_tool", "A file name doc_cache.md is text here. Answer literal.", False, {}, "literal", "file_search", "eval"),
    ]
    rows = []
    for task_id, family, prompt, needed, args, answer, distractor, split in base:
        rows.append({
            "task_id": task_id,
            "family": family,
            "user_prompt": prompt,
            "required_tool": family if needed else "none",
            "tool_needed": needed,
            "tool_args": args,
            "answer": answer,
            "distractor_tool": distractor,
            "split": split,
            "notes": "builtin Tier A smoke fallback",
            "surface_cues": {},
        })
    return rows


def task_from_row(row: Mapping[str, Any]) -> ToolTask:
    try:
        tool_args = dict(row.get("tool_args", {}) or {})
    except Exception as exc:
        raise ValueError(f"{row.get('task_id', '<missing>')}: tool_args must be a JSON object") from exc
    cues = row.get("surface_cues", {}) or {}
    if not isinstance(cues, dict):
        cues = {"raw_surface_cues": str(cues)}
    return ToolTask(
        task_id=str(row.get("task_id", "")).strip(),
        family=str(row.get("family", "")).strip(),
        user_prompt=str(row.get("user_prompt", "")).strip(),
        required_tool=str(row.get("required_tool", "")).strip(),
        tool_needed=bool(row.get("tool_needed", False)),
        tool_args=tool_args,
        answer=str(row.get("answer", "")).strip(),
        distractor_tool=str(row.get("distractor_tool", "")).strip(),
        split=str(row.get("split", "train")).strip().lower(),
        notes=str(row.get("notes", "")).strip(),
        surface_cues=cues,
    )


def validate_task_schema(task: ToolTask) -> list[str]:
    problems: list[str] = []
    if not task.task_id:
        problems.append("missing_task_id")
    if task.required_tool not in TOOLS:
        problems.append(f"required_tool_not_allowed:{task.required_tool}")
    if task.distractor_tool not in TOOLS:
        problems.append(f"distractor_tool_not_allowed:{task.distractor_tool}")
    if task.required_tool == task.distractor_tool:
        problems.append("distractor_equals_required_tool")
    if not task.user_prompt:
        problems.append("empty_user_prompt")
    if not task.answer:
        problems.append("empty_answer")
    if task.split not in {"train", "eval", "heldout", "test"}:
        problems.append(f"bad_split:{task.split}")
    if task.tool_needed != (task.required_tool != "none"):
        problems.append("tool_needed_disagrees_with_required_tool")
    return problems


def apply_caps(tasks: list[ToolTask], args: Any) -> list[ToolTask]:
    prompt_set = str(getattr(args, "prompt_set", "") or "")
    cap = PROMPT_SET_CAPS.get(prompt_set, 0)
    max_examples = int(getattr(args, "max_examples", 0) or 0)
    if max_examples > 0:
        cap = max_examples
    if not cap or len(tasks) <= cap:
        return sorted(tasks, key=task_sort_key)

    by_tool_split: dict[str, dict[str, list[ToolTask]]] = defaultdict(lambda: defaultdict(list))
    for task in sorted(tasks, key=task_sort_key):
        by_tool_split[task.required_tool][task.split].append(task)

    # Build each tool roster by interleaving train/eval rows. Without this,
    # Tier A caps can accidentally select train-only rows because frozen data is
    # grouped for human readability. The lab needs eval rows even in smoke mode
    # so split-aware artifacts do not become hollow parchment.
    grouped: dict[str, list[ToolTask]] = {}
    for tool in TOOLS:
        per_split = by_tool_split.get(tool, {})
        indices = {split: 0 for split in ("train", "eval", "heldout", "test")}
        roster: list[ToolTask] = []
        while True:
            progressed = False
            for split in ("train", "eval", "heldout", "test"):
                rows_for_split = per_split.get(split, [])
                idx = indices[split]
                if idx < len(rows_for_split):
                    roster.append(rows_for_split[idx])
                    indices[split] += 1
                    progressed = True
            if not progressed:
                break
        grouped[tool] = roster

    selected: list[ToolTask] = []
    cursor = 0
    while len(selected) < cap:
        progressed = False
        for tool in TOOLS:
            if cursor < len(grouped.get(tool, [])):
                selected.append(grouped[tool][cursor])
                progressed = True
                if len(selected) >= cap:
                    break
        if not progressed:
            break
        cursor += 1
    return sorted(selected, key=task_sort_key)


def load_tasks(ctx: bench.RunContext) -> tuple[list[ToolTask], dict[str, Any]]:
    path = data_path(ctx.args)
    source = "frozen_jsonl"
    if path.exists():
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        digest = file_sha256(path)
    else:
        if str(getattr(ctx.args, "tier", "")).lower() != "a":
            raise FileNotFoundError(f"Lab 34 task file not found: {path}. Tier B/C science runs require the frozen JSONL.")
        print("[lab34] frozen JSONL missing; using builtin Tier A smoke fallback. Do not ledger science claims from this run.")
        rows = builtin_smoke_rows()
        source = "builtin_tier_a_smoke_fallback"
        digest = hashlib.sha256("\n".join(row["task_id"] for row in rows).encode("utf-8")).hexdigest()

    tasks = [task_from_row(row) for row in rows]
    schema_rows = []
    kept: list[ToolTask] = []
    seen_ids: set[str] = set()
    for task in tasks:
        problems = validate_task_schema(task)
        if task.task_id in seen_ids:
            problems.append("duplicate_task_id")
        seen_ids.add(task.task_id)
        schema_rows.append({
            "task_id": task.task_id,
            "required_tool": task.required_tool,
            "tool_needed": task.tool_needed,
            "split": task.split,
            "kept": not problems,
            "problems": ";".join(problems),
        })
        if not problems:
            kept.append(task)
    if not kept:
        raise RuntimeError("Lab 34 schema validation dropped every task.")
    selected = apply_caps(kept, ctx.args)
    info = {
        "data_path": str(path),
        "sha256": digest,
        "data_source": source,
        "science_ready_data": source == "frozen_jsonl",
        "n_rows_file": len(tasks),
        "n_rows_schema_valid": len(kept),
        "n_rows_selected": len(selected),
        "families": dict(Counter(t.family for t in selected)),
        "required_tools": dict(Counter(t.required_tool for t in selected)),
        "splits": dict(Counter(t.split for t in selected)),
        "prompt_set": getattr(ctx.args, "prompt_set", ""),
        "max_examples": getattr(ctx.args, "max_examples", 0),
        "science_ready": source == "frozen_jsonl" and len(selected) >= SCIENCE_READY_MIN_ROWS and {"train", "eval"}.issubset({t.split for t in selected}),
        "safety_scope": "benign toy tools only; no web browsing, credentials, real filesystem writes, real calendar or file access, or harmful tools",
    }
    schema_path = ctx.path("diagnostics", "data_schema_validation.csv")
    bench.write_csv_with_context(ctx, schema_path, schema_rows)
    ctx.register_artifact(schema_path, "diagnostic", "Raw task schema validation before prompt-set caps.")
    return selected, info


def render_prompt(task: ToolTask) -> str:
    return (
        "You are in a controlled toy-tool harness.\n"
        "Available tools: calculator, dictionary, calendar, file_search, route_planner, unit_converter, or none.\n"
        "Choose based on the user's task, not on surface words alone.\n"
        f"User: {task.user_prompt.strip()}\n"
        "Next action state:"
    )


def intervention_prompt(task: ToolTask) -> str:
    return (
        "Choose the best next action for this controlled toy task.\n"
        "A=calculator B=dictionary C=calendar D=file_search E=route_planner F=unit_converter N=no_tool\n"
        f"User: {task.user_prompt.strip()}\n"
        "Action letter:"
    )


def letter_token_gate(ctx: bench.RunContext, bundle: bench.ModelBundle) -> dict[str, int]:
    rows: list[dict[str, Any]] = []
    ids: dict[str, int] = {}
    problems: list[str] = []
    for tool, letter in TOOL_LETTERS.items():
        token_text = " " + letter
        token_ids = bundle.tokenizer.encode(token_text, add_special_tokens=False)
        ok = len(token_ids) == 1
        if ok:
            ids[tool] = int(token_ids[0])
        else:
            problems.append(f"{tool}:{letter}:token_count={len(token_ids)}")
        rows.append({
            "tool": tool,
            "letter": letter,
            "token_text": token_text,
            "token_ids": " ".join(str(x) for x in token_ids),
            "token_count": len(token_ids),
            "single_token": ok,
        })
    path = ctx.path("diagnostics", "action_letter_token_gate.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic", "Single-token validation for constrained action-letter intervention.")
    if problems:
        raise RuntimeError("Lab 34 action-letter token gate failed: " + "; ".join(problems))
    return ids


def tokenization_gate(ctx: bench.RunContext, bundle: bench.ModelBundle, tasks: list[ToolTask]) -> tuple[list[ToolTask], dict[str, int], list[dict[str, Any]]]:
    letter_ids = letter_token_gate(ctx, bundle)
    rows: list[dict[str, Any]] = []
    kept: list[ToolTask] = []
    for task in tasks:
        problems = validate_task_schema(task)
        probe_ids = bundle.tokenizer(render_prompt(task), add_special_tokens=True)["input_ids"]
        action_ids = bundle.tokenizer(intervention_prompt(task), add_special_tokens=True)["input_ids"]
        if not probe_ids:
            problems.append("empty_probe_prompt_tokens")
        if not action_ids:
            problems.append("empty_action_prompt_tokens")
        if not problems:
            kept.append(task)
        rows.append({
            "task_id": task.task_id,
            "required_tool": task.required_tool,
            "split": task.split,
            "probe_prompt_tokens": len(probe_ids),
            "action_prompt_tokens": len(action_ids),
            "probe_final_token": bundle.tokenizer.decode([probe_ids[-1]]) if probe_ids else "",
            "action_final_token": bundle.tokenizer.decode([action_ids[-1]]) if action_ids else "",
            "kept": not problems,
            "problems": ";".join(problems),
            "probe_prompt_tail": render_prompt(task)[-160:].replace("\n", "\\n"),
        })
    path = ctx.path("diagnostics", "tokenization_gate.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic", "Prompt tokenization and action-letter validation for Lab 34.")
    if not kept:
        raise RuntimeError("Lab 34 tokenization gate dropped every task.")
    return kept, letter_ids, rows


# ---------------------------------------------------------------------------
# Surface controls
# ---------------------------------------------------------------------------


def prompt_features(task: ToolTask) -> dict[str, float]:
    text = task.user_prompt.lower()
    tool_name_count = sum(1 for tool in TOOLS if tool != "none" and tool in text)
    return {
        "length_chars": float(len(text)),
        "word_count": float(len(re.findall(r"[a-zA-Z0-9_./+*-]+", text))),
        "digit_count": float(sum(ch.isdigit() for ch in text)),
        "operator_count": float(sum(ch in "+-*/" for ch in text)),
        "tool_name_count": float(tool_name_count),
        "distractor_tool_mentioned": float(task.distractor_tool != "none" and task.distractor_tool in text),
        "lookup_word_count": float(sum(w in text for w in ("look up", "lookup", "glossary", "define", "definition"))),
        "calendar_word_count": float(sum(w in text for w in ("calendar", "scheduled", "event", "monday", "friday"))),
        "file_word_count": float(sum(w in text for w in ("doc", "document", "file", "search", "synthetic"))),
        "route_word_count": float(sum(w in text for w in ("route", "path", "graph", "->"))),
        "unit_word_count": float(sum(w in text for w in ("convert", "conversion", "miles", "kilometers", "pounds", "kilograms", "celsius", "fahrenheit", "minutes", "hours"))),
        "no_tool_phrase_count": float(sum(w in text for w in ("do not", "no lookup", "not require", "irrelevant", "appears here", "mentioned as text"))),
    }


def surface_tool_prediction(task: ToolTask) -> str:
    text = task.user_prompt.lower()
    if any(w in text for w in ("do not", "no lookup", "not require", "irrelevant", "mentioned as text", "appears here", "as examples")):
        # This is a generous surface baseline: it is allowed to notice explicit no-tool language.
        return "none"
    if re.search(r"\d+\s*[\+\-\*/]\s*\d+", text) or any(w in text for w in ("compute", "add", "subtract")):
        return "calculator"
    if any(w in text for w in ("dictionary", "glossary", "look up", "lookup", "define", "definition")):
        return "dictionary"
    if any(w in text for w in ("calendar", "scheduled", "standup", "demo prep", "bug triage", "design review")):
        return "calendar"
    if any(w in text for w in ("doc", "document", "file", "search", "synthetic")):
        return "file_search"
    if any(w in text for w in ("route", "path", "graph", "->")):
        return "route_planner"
    if any(w in text for w in ("convert", "conversion", "miles", "kilometers", "pounds", "kilograms", "celsius", "fahrenheit", "minutes", "hours")):
        return "unit_converter"
    return "none"


def surface_cue_audit_rows(tasks: Sequence[ToolTask]) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        feats = prompt_features(task)
        surface_pred = surface_tool_prediction(task)
        rows.append({
            "task_id": task.task_id,
            "required_tool": task.required_tool,
            "split": task.split,
            "surface_prediction": surface_pred,
            "surface_correct": surface_pred == task.required_tool,
            "surface_false_positive_no_tool": task.required_tool == "none" and surface_pred != "none",
            "surface_cues_json": json.dumps(task.surface_cues, sort_keys=True),
            **{k: rounded(v) for k, v in feats.items()},
        })
    return rows


# ---------------------------------------------------------------------------
# Deterministic toy tools
# ---------------------------------------------------------------------------


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
        if isinstance(sub, ast.Constant) and not isinstance(sub.value, (int, float)):
            raise ValueError("calculator constants must be numeric")

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


def run_tool(task: ToolTask) -> ToolExecution:
    try:
        if task.required_tool == "none":
            return ToolExecution("none", task.answer, tuple(), True)
        if task.required_tool == "calculator":
            value = safe_eval_expr(str(task.tool_args["expression"]))
            result = str(int(value)) if abs(value - int(value)) < 1e-9 else str(round(value, 4))
            return ToolExecution("calculator", result, ("arithmetic",), True)
        if task.required_tool == "dictionary":
            term = str(task.tool_args["term"])
            return ToolExecution("dictionary", GLOSSARY[term], (term,), True)
        if task.required_tool == "calendar":
            if "event" in task.tool_args:
                key = str(task.tool_args["event"])
                return ToolExecution("calendar", CALENDAR[key], (key,), True)
            if "after" in task.tool_args:
                return ToolExecution("calendar", "bug triage", ("standup", "bug triage"), True)
            if task.tool_args.get("day") == "Friday":
                return ToolExecution("calendar", "demo prep", ("Friday afternoon",), True)
            return ToolExecution("calendar", "demo prep", ("Friday afternoon",), True)
        if task.required_tool == "file_search":
            query = str(task.tool_args["query"]).lower()
            for name, text in DOCS.items():
                low = text.lower()
                if query in low or all(part in low for part in query.split()[:2]):
                    return ToolExecution("file_search", name, (name,), True)
            if "retry" in query or "backoff" in query:
                return ToolExecution("file_search", "doc_reliability.md", ("doc_reliability.md",), True)
            if "export" in query or "report" in query:
                return ToolExecution("file_search", "doc_export.md", ("doc_export.md",), True)
            if "cache" in query or "stale" in query:
                return ToolExecution("file_search", "doc_cache.md", ("doc_cache.md",), True)
            raise KeyError(query)
        if task.required_tool == "route_planner":
            start = str(task.tool_args["start"])
            end = str(task.tool_args["end"])
            return ToolExecution("route_planner", " -> ".join(shortest_path(start, end)), (start, end), True)
        if task.required_tool == "unit_converter":
            value = float(task.tool_args["value"])
            src = str(task.tool_args["from"])
            dst = str(task.tool_args["to"])
            if src == "miles" and dst == "kilometers":
                return ToolExecution("unit_converter", f"{value * 1.60934:.2f} kilometers", (src, dst), True)
            if src == "pounds" and dst == "kilograms":
                return ToolExecution("unit_converter", f"{value * 0.453592:.2f} kilograms", (src, dst), True)
            if src == "celsius" and dst == "fahrenheit":
                return ToolExecution("unit_converter", f"{int(value * 9 / 5 + 32)} Fahrenheit", (src, dst), True)
            if src == "hours" and dst == "minutes":
                return ToolExecution("unit_converter", f"{int(value * 60)} minutes", (src, dst), True)
        raise ValueError(task.required_tool)
    except Exception as exc:
        return ToolExecution(task.required_tool, "", tuple(), False, f"{type(exc).__name__}: {exc}")


def corrupted_result_for(task: ToolTask, result: str) -> str:
    if task.required_tool == "none":
        return task.answer
    if task.required_tool == "calculator":
        return "0" if result != "0" else "1"
    if task.required_tool == "dictionary":
        return "unrelated glossary definition"
    if task.required_tool == "calendar":
        return "Thursday 04:00"
    if task.required_tool == "file_search":
        return "doc_unrelated.md"
    if task.required_tool == "route_planner":
        return "A -> C"
    if task.required_tool == "unit_converter":
        return "999 units"
    return "CORRUPTED_RESULT"


def build_trace_tables(tasks: Sequence[ToolTask]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    trace_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    arg_rows: list[dict[str, Any]] = []
    for task in tasks:
        execn = run_tool(task)
        correct = execn.argument_valid and str(execn.result) == str(task.answer)
        corrupted = corrupted_result_for(task, execn.result)
        would_change = bool(task.required_tool != "none" and corrupted != task.answer)
        arg_rows.append({
            "argument_row_id": f"arg:{task.task_id}",
            "task_id": task.task_id,
            "required_tool": task.required_tool,
            "tool_args_json": json.dumps(task.tool_args, sort_keys=True),
            "argument_valid": execn.argument_valid,
            "error": execn.error,
            "result": execn.result,
            "expected_answer": task.answer,
            "result_matches_expected": correct,
        })
        trace_rows.append({
            "trace_id": f"trace:{task.task_id}",
            "task_id": task.task_id,
            "family": task.family,
            "split": task.split,
            "required_tool": task.required_tool,
            "tool_invoked": execn.tool_invoked,
            "tool_args_json": json.dumps(task.tool_args, sort_keys=True),
            "tool_result": execn.result,
            "expected_answer": task.answer,
            "result_matches_expected": correct,
            "memory_reads_json": json.dumps(list(execn.memory_reads)),
            "corrupted_tool_result": corrupted,
            "would_final_answer_change_if_tool_result_corrupted": would_change,
            "argument_valid": execn.argument_valid,
            "error": execn.error,
        })
        transition_rows.extend([
            {"task_id": task.task_id, "step_index": 0, "state": "user_prompt_received", "tool": "", "detail": task.user_prompt},
            {"task_id": task.task_id, "step_index": 1, "state": "oracle_required_tool", "tool": task.required_tool, "detail": json.dumps(task.tool_args, sort_keys=True)},
            {"task_id": task.task_id, "step_index": 2, "state": "local_tool_result", "tool": execn.tool_invoked, "detail": execn.result},
            {"task_id": task.task_id, "step_index": 3, "state": "final_answer", "tool": execn.tool_invoked, "detail": task.answer},
        ])
        report_rows.append({
            "self_report_row_id": f"self_report:{task.task_id}",
            "task_id": task.task_id,
            "required_tool": task.required_tool,
            "known_trace_label": execn.tool_invoked,
            "known_trace_args_json": json.dumps(task.tool_args, sort_keys=True),
            "known_trace_result": execn.result,
            "self_report_template": "no tool was needed" if execn.tool_invoked == "none" else f"used {execn.tool_invoked} with args {json.dumps(task.tool_args, sort_keys=True)}",
            "model_self_report_generated": False,
            "matches_known_trace": correct,
            "requires_human_review": True,
            "review_note": "Template row for source-attribution review; this is not model introspection.",
            "student_trace_label": "",
            "student_confidence": "",
            "student_evidence_span": "",
            "reviewer_trace_label": "",
            "agreement_status": "",
        })
    return trace_rows, transition_rows, report_rows, arg_rows


# ---------------------------------------------------------------------------
# Residual vectors and probe models
# ---------------------------------------------------------------------------


def choose_depths(bundle: bench.ModelBundle, prompt_set: str) -> list[int]:
    n = int(bundle.anatomy.n_layers)
    if prompt_set == "full":
        return list(range(n + 1))
    return sorted({0, max(1, n // 4), max(1, n // 2), max(1, (3 * n) // 4), n})


def capture_vectors(ctx: bench.RunContext, bundle: bench.ModelBundle, tasks: Sequence[ToolTask], depths: Sequence[int]) -> tuple[dict[tuple[str, int], Any], list[dict[str, Any]]]:
    vectors: dict[tuple[str, int], Any] = {}
    audit_rows: list[dict[str, Any]] = []
    report_every = max(1, len(tasks) // 4)
    for i, task in enumerate(tasks, start=1):
        prompt = render_prompt(task)
        capture = bench.run_with_residual_cache(bundle, prompt)
        for depth in depths:
            vectors[(task.task_id, int(depth))] = capture.streams[int(depth), -1, :].detach().clone()
        audit_rows.append({
            "task_id": task.task_id,
            "required_tool": task.required_tool,
            "split": task.split,
            "n_tokens": len(capture.input_ids),
            "final_token_text": capture.tokens_text[-1] if capture.tokens_text else "",
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            "prompt_tail": prompt[-180:].replace("\n", "\\n"),
        })
        if i % report_every == 0 or i == len(tasks):
            print(f"[lab34] captured prompt-boundary states {i}/{len(tasks)}")
    path = ctx.path("diagnostics", "prompt_boundary_audit.csv")
    bench.write_csv_with_context(ctx, path, audit_rows)
    ctx.register_artifact(path, "diagnostic", "Prompt-boundary token counts and final-token audit.")
    return vectors, audit_rows


def build_direction_model(ctx: bench.RunContext, bundle: bench.ModelBundle, tasks: Sequence[ToolTask], vectors: Mapping[tuple[str, int], Any], depth: int) -> DirectionModel:
    import torch

    train = [t for t in tasks if t.split == "train"] or list(tasks)
    needed_vecs = [vectors[(t.task_id, depth)] for t in train if t.tool_needed]
    none_vecs = [vectors[(t.task_id, depth)] for t in train if not t.tool_needed]
    if not needed_vecs or not none_vecs:
        raise RuntimeError("Lab 34 needs both tool-needed and no-tool train rows to build directions.")
    needed_dir = unit_vector(mean_vec(needed_vecs) - mean_vec(none_vecs))
    dirs: dict[str, Any] = {"tool_needed": needed_dir, "none": -needed_dir}
    all_train_vecs = [vectors[(t.task_id, depth)] for t in train]
    train_center = mean_vec(all_train_vecs)
    for tool in TOOLS:
        if tool == "none":
            continue
        pos = [vectors[(t.task_id, depth)] for t in train if t.required_tool == tool]
        neg = [vectors[(t.task_id, depth)] for t in train if t.required_tool != tool]
        dirs[tool] = unit_vector(mean_vec(pos) - mean_vec(neg)) if pos and neg else torch.zeros_like(needed_dir)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(getattr(ctx.args, "seed", 0) or 0) + 34000 + depth)
    for idx in range(N_RANDOM_INTERVENTION_CONTROLS):
        dirs[f"random_direction_control_{idx:02d}"] = unit_vector(torch.randn(needed_dir.shape, dtype=needed_dir.dtype, generator=gen))
    dirs["random_direction_control"] = dirs["random_direction_control_00"]
    needed_scores_train = [float(vectors[(t.task_id, depth)].float() @ dirs["tool_needed"].float()) for t in train]
    needed_labels_train = [1 if t.tool_needed else 0 for t in train]
    threshold = best_binary_threshold(needed_labels_train, needed_scores_train)
    centered_norm = safe_mean([float(torch.linalg.vector_norm(v.float() - train_center.float())) for v in all_train_vecs], default=1.0)
    steer = {name: vec.float() * float(centered_norm) for name, vec in dirs.items()}
    steer_layer = max(0, min(bundle.anatomy.n_layers - 1, int(depth) - 1))
    metadata = {
        "depth": int(depth),
        "steer_layer": steer_layer,
        "needed_threshold": threshold,
        "train_centered_norm": rounded(centered_norm),
        "n_train": len(train),
        "n_train_needed": sum(1 for t in train if t.tool_needed),
        "n_train_none": sum(1 for t in train if not t.tool_needed),
        "direction_norms": {name: rounded(float(torch.linalg.vector_norm(vec.float()))) for name, vec in dirs.items()},
        "steer_vector_norms": {name: rounded(float(torch.linalg.vector_norm(vec.float()))) for name, vec in steer.items()},
    }
    return DirectionModel(depth=depth, steer_layer=steer_layer, unit=dirs, steer=steer, needed_threshold=threshold, train_center_norm=float(centered_norm), metadata=metadata)


def score_tool_prediction(task: ToolTask, vec: Any, model: DirectionModel) -> dict[str, Any]:
    needed_score = float(vec.float() @ model.unit["tool_needed"].float())
    needed_pred = needed_score >= model.needed_threshold
    tool_scores = {tool: float(vec.float() @ model.unit[tool].float()) for tool in TOOLS if tool != "none"}
    sorted_tools = sorted(tool_scores, key=lambda t: tool_scores[t], reverse=True)
    best_tool = sorted_tools[0] if sorted_tools else "none"
    second_tool = sorted_tools[1] if len(sorted_tools) > 1 else "none"
    if not needed_pred:
        pred = "none"
        margin = needed_score - model.needed_threshold
    else:
        pred = best_tool
        margin = tool_scores[best_tool] - tool_scores.get(second_tool, 0.0)
    surface_pred = surface_tool_prediction(task)
    return {
        "needed_score": needed_score,
        "needed_prediction": bool(needed_pred),
        "tool_probe_prediction": pred,
        "tool_probe_correct": pred == task.required_tool,
        "tool_score_json": json.dumps({k: rounded(v) for k, v in tool_scores.items()}, sort_keys=True),
        "top_tool_score_margin": margin,
        "surface_cue_prediction": surface_pred,
        "surface_cue_correct": surface_pred == task.required_tool,
        "surface_needed_prediction": surface_pred != "none",
    }


def split_rows(tasks: Sequence[ToolTask], split: str) -> list[ToolTask]:
    if split == "all":
        return list(tasks)
    if split == "eval":
        return [t for t in tasks if t.split in {"eval", "heldout", "test"}]
    return [t for t in tasks if t.split == split]


def summarize_probe_split(tasks: Sequence[ToolTask], vectors: Mapping[tuple[str, int], Any], model: DirectionModel, split: str) -> dict[str, Any]:
    rows = split_rows(tasks, split)
    if not rows:
        return {
            "depth": model.depth,
            "split_group": split,
            "n_tasks": 0,
            "tool_needed_auc": "",
            "tool_needed_accuracy": "",
            "surface_needed_accuracy": "",
            "tool_selection_accuracy": "",
            "surface_control_accuracy": "",
            "shuffled_label_control_accuracy": "",
            "no_tool_false_positive_rate": "",
            "decode_gap_over_surface": "",
        }
    scored = [score_tool_prediction(t, vectors[(t.task_id, model.depth)], model) for t in rows]
    needed_labels = [1 if t.tool_needed else 0 for t in rows]
    needed_scores = [s["needed_score"] for s in scored]
    needed_acc = safe_mean([1.0 if bool(s["needed_prediction"]) == bool(y) else 0.0 for s, y in zip(scored, needed_labels)], 0.0)
    surface_needed_acc = safe_mean([1.0 if bool(s["surface_needed_prediction"]) == bool(y) else 0.0 for s, y in zip(scored, needed_labels)], 0.0)
    tool_acc = safe_mean([1.0 if s["tool_probe_prediction"] == t.required_tool else 0.0 for s, t in zip(scored, rows)], 0.0)
    surface_acc = safe_mean([1.0 if s["surface_cue_prediction"] == t.required_tool else 0.0 for s, t in zip(scored, rows)], 0.0)
    shuffled_labels = [rows[(i + max(1, len(rows) // 3)) % len(rows)].required_tool for i in range(len(rows))]
    shuffled_acc = safe_mean([1.0 if s["tool_probe_prediction"] == shuffled_labels[i] else 0.0 for i, s in enumerate(scored)], 0.0)
    no_tool_rows = [(t, s) for t, s in zip(rows, scored) if t.required_tool == "none"]
    no_tool_fp = safe_mean([1.0 if s["tool_probe_prediction"] != "none" else 0.0 for _t, s in no_tool_rows], 0.0 if no_tool_rows else float("nan"))
    return {
        "depth": model.depth,
        "split_group": split,
        "n_tasks": len(rows),
        "tool_needed_auc": rounded(auc_binary(needed_labels, needed_scores)),
        "tool_needed_accuracy": rounded(needed_acc),
        "surface_needed_accuracy": rounded(surface_needed_acc),
        "tool_selection_accuracy": rounded(tool_acc),
        "surface_control_accuracy": rounded(surface_acc),
        "shuffled_label_control_accuracy": rounded(shuffled_acc),
        "no_tool_false_positive_rate": rounded(no_tool_fp),
        "decode_gap_over_surface": rounded(tool_acc - surface_acc),
        "needed_gap_over_surface": rounded(needed_acc - surface_needed_acc),
        "claim_scope": "prompt_boundary_decode",
    }


def build_probe_reports(tasks: Sequence[ToolTask], vectors: Mapping[tuple[str, int], Any], models_by_depth: Mapping[int, DirectionModel]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for depth, model in sorted(models_by_depth.items()):
        for split in ("train", "eval", "all"):
            row = summarize_probe_split(tasks, vectors, model, split)
            row["probe_cell_id"] = f"depth{depth}:{split}"
            rows.append(row)
        train_row = next(r for r in rows if int(r["depth"]) == depth and r["split_group"] == "train")
        claimable = 0 < int(depth) < max(models_by_depth)
        selection_score = as_float(train_row.get("decode_gap_over_surface"), -999.0) + 0.25 * (as_float(train_row.get("tool_needed_auc"), 0.5) - 0.5)
        selection_rows.append({
            "selection_cell_id": f"depth{depth}:train_selection",
            "depth": depth,
            "claimable_depth": claimable,
            "train_tool_needed_auc": train_row.get("tool_needed_auc"),
            "train_tool_selection_accuracy": train_row.get("tool_selection_accuracy"),
            "train_surface_control_accuracy": train_row.get("surface_control_accuracy"),
            "train_decode_gap_over_surface": train_row.get("decode_gap_over_surface"),
            "selection_score": rounded(selection_score),
        })
    return rows, selection_rows


def select_depth(selection_rows: Sequence[Mapping[str, Any]]) -> int:
    candidates = [r for r in selection_rows if r.get("claimable_depth")]
    if not candidates:
        candidates = list(selection_rows)
    if not candidates:
        raise RuntimeError("No depth selection rows available.")
    chosen = max(
        candidates,
        key=lambda r: (
            as_float(r.get("selection_score"), -999.0),
            as_float(r.get("train_tool_needed_auc"), -999.0),
            as_float(r.get("train_tool_selection_accuracy"), -999.0),
            -int(r.get("depth", 999)),
        ),
    )
    return int(chosen["depth"])


def task_manifest_rows(tasks: Sequence[ToolTask], vectors: Mapping[tuple[str, int], Any], model: DirectionModel) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        scored = score_tool_prediction(task, vectors[(task.task_id, model.depth)], model)
        feats = prompt_features(task)
        rows.append({
            "tool_task_row_id": f"task:{task.task_id}:depth{model.depth}",
            "task_id": task.task_id,
            "family": task.family,
            "split": task.split,
            "required_tool": task.required_tool,
            "tool_needed": task.tool_needed,
            "distractor_tool": task.distractor_tool,
            "selected_depth": model.depth,
            "user_prompt": task.user_prompt,
            "answer": task.answer,
            "notes": task.notes,
            "tool_probe_prediction": scored["tool_probe_prediction"],
            "tool_probe_correct": scored["tool_probe_correct"],
            "needed_score": rounded(scored["needed_score"]),
            "needed_threshold": rounded(model.needed_threshold),
            "needed_prediction": scored["needed_prediction"],
            "top_tool_score_margin": rounded(scored["top_tool_score_margin"]),
            "tool_score_json": scored["tool_score_json"],
            "surface_cue_prediction": scored["surface_cue_prediction"],
            "surface_cue_correct": scored["surface_cue_correct"],
            "surface_cues_json": json.dumps(task.surface_cues, sort_keys=True),
            **{k: rounded(v) for k, v in feats.items()},
            "student_trace_label": "",
            "student_confidence": "",
            "student_evidence_span": "",
            "reviewer_trace_label": "",
            "agreement_status": "",
        })
    return rows


def confusion_rows(task_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("eval", "train", "all"):
        selected = [r for r in task_rows if split == "all" or r.get("split") == split]
        counts = Counter((str(r["required_tool"]), str(r["tool_probe_prediction"])) for r in selected)
        for required in TOOLS:
            for pred in TOOLS:
                count = counts.get((required, pred), 0)
                if count:
                    rows.append({"split_group": split, "required_tool": required, "predicted_tool": pred, "count": count})
    return rows


# ---------------------------------------------------------------------------
# Causal action-letter intervention
# ---------------------------------------------------------------------------


def next_token_logits_raw(bundle: bench.ModelBundle, prompt: str, *, steer: tuple[int, Any, float] | None = None) -> Any:
    import contextlib
    import torch

    encoded = bundle.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(bundle.input_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.input_device)
    cm = bench.steering_hooks(bundle, steer[0], steer[1], steer[2]) if steer is not None else contextlib.nullcontext()
    with cm, torch.no_grad():
        out = bundle.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return bench.tensor_cpu_float(out.logits[0, -1])


def run_interventions(
    bundle: bench.ModelBundle,
    tasks: Sequence[ToolTask],
    model: DirectionModel,
    letter_ids: Mapping[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for i, task in enumerate(tasks, start=1):
        prompt = intervention_prompt(task)
        target_tool = task.required_tool
        distractor_tool = task.distractor_tool if task.distractor_tool != target_tool else ("none" if target_tool != "none" else "calculator")
        target_id = int(letter_ids[target_tool])
        distractor_id = int(letter_ids[distractor_tool])
        random_names = sorted(name for name in model.steer if name.startswith("random_direction_control_"))
        direction_entries = [
            ("target_tool_direction", target_tool, model.steer.get(target_tool, model.steer["none"])),
            ("tool_needed_direction", "tool_needed", model.steer["tool_needed"]),
        ]
        direction_entries.extend(
            ("random_direction_control", random_name, model.steer[random_name])
            for random_name in random_names
        )
        task_row_start = len(rows)
        base_by_direction: dict[tuple[str, str], float] = {}
        for condition, direction_name, vector in direction_entries:
            for scale in STEER_SCALES:
                logits = next_token_logits_raw(bundle, prompt, steer=(model.steer_layer, vector, float(scale)))
                margin = float(logits[target_id] - logits[distractor_id])
                if abs(float(scale)) < 1e-12:
                    base_by_direction[(condition, direction_name)] = margin
                rows.append({
                    "intervention_id": f"{task.task_id}:{condition}:scale_{float(scale):g}",
                    "direction_id": f"depth{model.depth}:{direction_name}",
                    "task_id": task.task_id,
                    "family": task.family,
                    "split": task.split,
                    "required_tool": task.required_tool,
                    "distractor_tool": distractor_tool,
                    "condition": condition,
                    "direction": direction_name,
                    "scale": float(scale),
                    "selected_depth": model.depth,
                    "steer_layer": model.steer_layer,
                    "target_letter": TOOL_LETTERS[target_tool],
                    "target_letter_token_id": target_id,
                    "distractor_letter": TOOL_LETTERS[distractor_tool],
                    "distractor_letter_token_id": distractor_id,
                    "target_minus_distractor_logit": rounded(margin),
                    "shift_from_condition_zero": "",  # filled below
                    "prompt": prompt,
                })
        for row in rows[task_row_start:]:
            base = base_by_direction.get((str(row["condition"]), str(row["direction"])))
            if base is not None:
                row["shift_from_condition_zero"] = rounded(as_float(row["target_minus_distractor_logit"]) - base)
        if i % max(1, len(tasks) // 4) == 0 or i == len(tasks):
            print(f"[lab34] action-letter interventions {i}/{len(tasks)}")
    summary = intervention_summary_rows(rows)
    return rows, summary


def intervention_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ("train", "eval", "all"):
        selected = [r for r in rows if split == "all" or r.get("split") == split]
        for condition in sorted({str(r["condition"]) for r in selected}):
            for scale in sorted({float(r["scale"]) for r in selected}):
                subset = [r for r in selected if str(r["condition"]) == condition and float(r["scale"]) == scale]
                out.append({
                    "split_group": split,
                    "condition": condition,
                    "scale": scale,
                    "n": len(subset),
                    "mean_target_minus_distractor_logit": rounded(safe_mean([r.get("target_minus_distractor_logit") for r in subset])),
                    "mean_shift_from_zero": rounded(safe_mean([r.get("shift_from_condition_zero") for r in subset])),
                    "stdev_shift_from_zero": rounded(safe_stdev([r.get("shift_from_condition_zero") for r in subset])),
                })
    return out


def summary_value(summary_rows: Sequence[Mapping[str, Any]], split: str, condition: str, scale: float, key: str) -> float:
    for row in summary_rows:
        if row.get("split_group") == split and row.get("condition") == condition and abs(float(row.get("scale", 999)) - float(scale)) < 1e-9:
            return as_float(row.get(key))
    return float("nan")


# ---------------------------------------------------------------------------
# Evidence, counterexamples, and reports
# ---------------------------------------------------------------------------


def selected_probe_row(probe_rows: Sequence[Mapping[str, Any]], selected_depth: int, split: str = "eval") -> Mapping[str, Any]:
    for row in probe_rows:
        if int(row.get("depth", -1)) == int(selected_depth) and row.get("split_group") == split:
            return row
    for row in probe_rows:
        if int(row.get("depth", -1)) == int(selected_depth) and row.get("split_group") == "all":
            return row
    return {}


def build_evidence_matrix(
    data_info: Mapping[str, Any],
    selected_depth: int,
    probe_rows: Sequence[Mapping[str, Any]],
    intervention_summary: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    self_report_rows: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eval_probe = selected_probe_row(probe_rows, selected_depth, "eval")
    eval_split = "eval" if eval_probe and int(eval_probe.get("n_tasks", 0) or 0) else "all"
    if eval_split == "all":
        eval_probe = selected_probe_row(probe_rows, selected_depth, "all")
    needed_auc = as_float(eval_probe.get("tool_needed_auc"))
    needed_acc = as_float(eval_probe.get("tool_needed_accuracy"))
    surface_needed = as_float(eval_probe.get("surface_needed_accuracy"))
    selection_acc = as_float(eval_probe.get("tool_selection_accuracy"))
    surface_acc = as_float(eval_probe.get("surface_control_accuracy"))
    shuffled_acc = as_float(eval_probe.get("shuffled_label_control_accuracy"))
    decode_gap = selection_acc - surface_acc if math.isfinite(selection_acc) and math.isfinite(surface_acc) else float("nan")
    needed_gap = needed_acc - surface_needed if math.isfinite(needed_acc) and math.isfinite(surface_needed) else float("nan")
    target_shift = summary_value(intervention_summary, eval_split, "target_tool_direction", CLAIMABLE_SCALE, "mean_shift_from_zero")
    random_shift = summary_value(intervention_summary, eval_split, "random_direction_control", CLAIMABLE_SCALE, "mean_shift_from_zero")
    causal_gap = target_shift - random_shift if math.isfinite(target_shift) and math.isfinite(random_shift) else float("nan")
    trace_match_rate = safe_mean([1.0 if row.get("result_matches_expected") else 0.0 for row in trace_rows], 0.0)
    arg_valid_rate = safe_mean([1.0 if row.get("argument_valid") else 0.0 for row in trace_rows], 0.0)
    review_required_rate = safe_mean([1.0 if row.get("requires_human_review") else 0.0 for row in self_report_rows], 0.0)

    needed_supported = math.isfinite(needed_auc) and needed_auc >= DECODE_AUC_BAR and (not math.isfinite(needed_gap) or needed_gap >= -0.02)
    selection_supported = math.isfinite(decode_gap) and decode_gap >= DECODE_GAP_BAR and selection_acc >= max(shuffled_acc, 0.0)
    causal_supported = math.isfinite(causal_gap) and causal_gap >= CAUSAL_GAP_BAR and target_shift > 0.0
    trace_supported = trace_match_rate >= 0.999 and arg_valid_rate >= 0.999

    evidence = [
        {
            "method": "prompt_boundary_tool_needed_decode",
            "evidence_rung": "DECODE",
            "selected_depth": selected_depth,
            "eval_split_used": eval_split,
            "metric": "tool_needed_auc",
            "value": rounded(needed_auc),
            "control_metric": "surface_needed_accuracy",
            "control_value": rounded(surface_needed),
            "gap_over_control": rounded(needed_gap),
            "claim_posture": "supported" if needed_supported else "surface_confounded_or_weak",
            "allowed_claim_fragment": "tool-needed state is decodable above controls" if needed_supported else "tool-needed evidence needs refinement",
        },
        {
            "method": "prompt_boundary_tool_selection_decode",
            "evidence_rung": "DECODE",
            "selected_depth": selected_depth,
            "eval_split_used": eval_split,
            "metric": "tool_selection_accuracy",
            "value": rounded(selection_acc),
            "control_metric": "surface_control_accuracy",
            "control_value": rounded(surface_acc),
            "gap_over_control": rounded(decode_gap),
            "claim_posture": "supported" if selection_supported else "surface_confounded_or_weak",
            "allowed_claim_fragment": "which-tool state is decodable above surface controls" if selection_supported else "which-tool evidence is not above the surface baseline",
        },
        {
            "method": "constrained_action_letter_activation_addition",
            "evidence_rung": "CAUSAL",
            "selected_depth": selected_depth,
            "eval_split_used": eval_split,
            "metric": "target_direction_shift_at_scale_1",
            "value": rounded(target_shift),
            "control_metric": "random_direction_shift_at_scale_1",
            "control_value": rounded(random_shift),
            "gap_over_control": rounded(causal_gap),
            "claim_posture": "supported_narrow_letter_prompt" if causal_supported else "random_or_letter_prompt_limited",
            "allowed_claim_fragment": "activation addition shifted constrained tool-choice logits" if causal_supported else "activation addition did not beat random control clearly",
        },
        {
            "method": "deterministic_local_tool_trace",
            "evidence_rung": "OBS+AUDIT",
            "selected_depth": selected_depth,
            "eval_split_used": "all",
            "metric": "result_match_rate",
            "value": rounded(trace_match_rate),
            "control_metric": "argument_valid_rate",
            "control_value": rounded(arg_valid_rate),
            "gap_over_control": "",
            "claim_posture": "trace_validated" if trace_supported else "trace_or_argument_mismatch",
            "allowed_claim_fragment": "local toy-tool trace matched expected answers" if trace_supported else "fix tool/data mismatch before claims",
        },
        {
            "method": "tool_self_report_review_scaffold",
            "evidence_rung": "SELF-REPORT",
            "selected_depth": selected_depth,
            "eval_split_used": "all",
            "metric": "requires_human_review_rate",
            "value": rounded(review_required_rate),
            "control_metric": "model_self_report_generated",
            "control_value": 0.0,
            "gap_over_control": "",
            "claim_posture": "review_required_not_introspection",
            "allowed_claim_fragment": "known trace labels are review scaffolds, not model introspection",
        },
    ]
    metrics = {
        "lab_id": LAB_ID,
        "lab_name": LAB_NAME,
        "selected_depth": selected_depth,
        "eval_split_used": eval_split,
        "tool_needed_auc": rounded(needed_auc),
        "tool_needed_accuracy": rounded(needed_acc),
        "surface_needed_accuracy": rounded(surface_needed),
        "tool_selection_accuracy": rounded(selection_acc),
        "surface_control_accuracy": rounded(surface_acc),
        "shuffled_label_control_accuracy": rounded(shuffled_acc),
        "decode_gap_over_surface": rounded(decode_gap),
        "target_direction_shift_at_scale_1": rounded(target_shift),
        "random_direction_shift_at_scale_1": rounded(random_shift),
        "causal_shift_over_random": rounded(causal_gap),
        "trace_match_rate": rounded(trace_match_rate),
        "argument_valid_rate": rounded(arg_valid_rate),
        "n_counterexamples": len(counterexamples),
        "data": data_info,
        "supported": {
            "tool_needed_decode": needed_supported,
            "tool_selection_decode": selection_supported,
            "causal_action_letter_shift": causal_supported,
            "trace_validated": trace_supported,
            "self_report_review_required": True,
        },
    }
    return evidence, metrics


def build_counterexamples(
    task_rows: Sequence[Mapping[str, Any]],
    intervention_rows: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in task_rows:
        split = str(row.get("split"))
        if split not in {"eval", "heldout", "test"}:
            continue
        required = str(row.get("required_tool"))
        pred = str(row.get("tool_probe_prediction"))
        surface = str(row.get("surface_cue_prediction"))
        severity = abs(as_float(row.get("top_tool_score_margin"), 0.0))
        if required == "none" and pred != "none":
            rows.append({
                "kind": "no_tool_surface_cue_false_positive",
                "severity": rounded(max(0.0, severity)),
                "task_id": row["task_id"],
                "split": split,
                "required_tool": required,
                "probe_prediction": pred,
                "surface_prediction": surface,
                "user_prompt": row.get("user_prompt", ""),
                "why_it_matters": "No-tool rows with tool words are the main guard against intention language.",
            })
        elif pred != required and surface == required:
            rows.append({
                "kind": "surface_beats_probe",
                "severity": rounded(max(0.0, severity)),
                "task_id": row["task_id"],
                "split": split,
                "required_tool": required,
                "probe_prediction": pred,
                "surface_prediction": surface,
                "user_prompt": row.get("user_prompt", ""),
                "why_it_matters": "The surface baseline explained the row better than the residual probe.",
            })
        elif pred != required:
            rows.append({
                "kind": "probe_tool_confusion",
                "severity": rounded(max(0.0, severity)),
                "task_id": row["task_id"],
                "split": split,
                "required_tool": required,
                "probe_prediction": pred,
                "surface_prediction": surface,
                "user_prompt": row.get("user_prompt", ""),
                "why_it_matters": "Which-tool decoding failed on a held-out task.",
            })
    by_task_condition: dict[tuple[str, str], dict[float, float]] = defaultdict(dict)
    for row in intervention_rows:
        by_task_condition[(str(row["task_id"]), str(row["condition"]))][float(row["scale"])] = as_float(row.get("shift_from_condition_zero"))
    for (task_id, _condition), vals in list(by_task_condition.items()):
        target = by_task_condition.get((task_id, "target_tool_direction"), {})
        random = by_task_condition.get((task_id, "random_direction_control"), {})
        t = target.get(CLAIMABLE_SCALE)
        r = random.get(CLAIMABLE_SCALE)
        if t is not None and r is not None and math.isfinite(t) and math.isfinite(r) and t <= r:
            rows.append({
                "kind": "random_direction_matches_or_beats_target_direction",
                "severity": rounded(r - t),
                "task_id": task_id,
                "split": next((str(x.get("split")) for x in intervention_rows if x.get("task_id") == task_id), ""),
                "required_tool": next((str(x.get("required_tool")) for x in intervention_rows if x.get("task_id") == task_id), ""),
                "probe_prediction": "",
                "surface_prediction": "",
                "user_prompt": next((str(x.get("prompt")) for x in intervention_rows if x.get("task_id") == task_id), ""),
                "why_it_matters": "The causal letter-prompt test is not specific if random shifts as much as the target direction.",
            })
    for row in trace_rows:
        if not row.get("result_matches_expected"):
            rows.append({
                "kind": "tool_trace_mismatch",
                "severity": 1.0,
                "task_id": row["task_id"],
                "split": row.get("split", ""),
                "required_tool": row.get("required_tool", ""),
                "probe_prediction": "",
                "surface_prediction": "",
                "user_prompt": "",
                "why_it_matters": "The deterministic toy tool did not reproduce the frozen expected answer.",
            })
    rows.sort(key=lambda r: as_float(r.get("severity"), 0.0), reverse=True)
    return rows[:40]


# ---------------------------------------------------------------------------
# Visualization and artifact quality helpers
# ---------------------------------------------------------------------------


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(value)


def safe_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value or "[]"))
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def finite_pairs(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        val = as_float(row.get(key))
        if math.isfinite(val):
            vals.append(val)
    return vals


def split_for_fig(rows: Sequence[Mapping[str, Any]], *, split_key: str = "split") -> str:
    splits = {str(row.get(split_key)) for row in rows}
    return "eval" if "eval" in splits else "all"


def write_figure_source(ctx: bench.RunContext, filename: str, rows: Sequence[Mapping[str, Any]], description: str) -> str:
    path = ctx.path("tables", "figure_sources", filename)
    bench.write_csv_with_context(ctx, path, list(rows))
    ctx.register_artifact(path, "table", description)
    return str(path.relative_to(ctx.run_dir))


def write_plot_manifest(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = {
        "lab": LAB_NAME,
        "note": "Every plot row names its source table. Claims should be checked against the tables before being copied into the ledger.",
        "figures": list(rows),
    }
    json_path = ctx.path("plots", "plot_manifest.json")
    bench.write_json(json_path, payload)
    ctx.register_artifact(json_path, "plot_manifest", "Figure manifest with source tables, row counts, metrics, controls, and claim boundaries.")
    csv_path = ctx.path("plots", "plot_manifest.csv")
    bench.write_csv_with_context(ctx, csv_path, list(rows))
    ctx.register_artifact(csv_path, "plot_manifest", "CSV copy of the Lab 34 figure manifest.")


def add_empty_message(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def source_target_vs_control(task_rows: Sequence[Mapping[str, Any]], intervention_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split = "eval" if any(str(r.get("split")) == "eval" for r in task_rows) else "all"
    for row in task_rows:
        if split != "all" and str(row.get("split")) != split:
            continue
        rows.append({
            "source_part": "probe_vs_surface",
            "task_id": row.get("task_id"),
            "split": row.get("split"),
            "required_tool": row.get("required_tool"),
            "target_metric": "probe_correct",
            "target_value": 1.0 if boolish(row.get("tool_probe_correct")) else 0.0,
            "control_metric": "surface_correct",
            "control_value": 1.0 if boolish(row.get("surface_cue_correct")) else 0.0,
            "difference_target_minus_control": (1.0 if boolish(row.get("tool_probe_correct")) else 0.0) - (1.0 if boolish(row.get("surface_cue_correct")) else 0.0),
            "probe_prediction": row.get("tool_probe_prediction"),
            "surface_prediction": row.get("surface_cue_prediction"),
            "sample_count": 1,
        })
    task_ids = sorted({str(r.get("task_id")) for r in intervention_rows})
    for task_id in task_ids:
        if not task_id:
            continue
        target = next((r for r in intervention_rows if str(r.get("task_id")) == task_id and str(r.get("condition")) == "target_tool_direction" and abs(as_float(r.get("scale")) - CLAIMABLE_SCALE) < 1e-9), None)
        random = next((r for r in intervention_rows if str(r.get("task_id")) == task_id and str(r.get("condition")) == "random_direction_control" and abs(as_float(r.get("scale")) - CLAIMABLE_SCALE) < 1e-9), None)
        if target is None or random is None:
            continue
        rows.append({
            "source_part": "activation_target_vs_random",
            "task_id": task_id,
            "split": target.get("split"),
            "required_tool": target.get("required_tool"),
            "target_metric": "target_direction_shift_at_scale_1",
            "target_value": as_float(target.get("shift_from_condition_zero")),
            "control_metric": "random_direction_shift_at_scale_1",
            "control_value": as_float(random.get("shift_from_condition_zero")),
            "difference_target_minus_control": as_float(target.get("shift_from_condition_zero")) - as_float(random.get("shift_from_condition_zero")),
            "probe_prediction": "",
            "surface_prediction": "",
            "sample_count": 1,
        })
    return rows


def write_run_config_snapshot(ctx: bench.RunContext, data_info: Mapping[str, Any], model: DirectionModel, depths: Sequence[int]) -> dict[str, Any]:
    payload = {
        "lab": LAB_NAME,
        "model_id": ctx.model_id,
        "model_revision": ctx.model_revision,
        "tier": ctx.args.tier,
        "dtype": ctx.args.dtype,
        "quantization": ctx.args.quantization,
        "prompt_set": ctx.args.prompt_set,
        "max_examples": ctx.args.max_examples,
        "seed": ctx.args.seed,
        "decoding_settings": {"do_generation": False, "causal_prompt": "single next-token action-letter logits"},
        "direction_settings": {
            "depths_scanned": list(depths),
            "selected_depth": model.depth,
            "steer_layer": model.steer_layer,
            "claimable_scale": CLAIMABLE_SCALE,
            "steer_scales": list(STEER_SCALES),
            "needed_threshold": model.needed_threshold,
        },
        "data": dict(data_info),
        "surface_control": "deterministic lexical heuristic over prompt features",
        "safety_scope": data_info.get("safety_scope"),
    }
    path = ctx.path("diagnostics", "lab34_run_config_snapshot.json")
    bench.write_json(path, payload)
    ctx.register_artifact(path, "diagnostic", "Run config snapshot for reproducing Lab 34 plots and tables.")
    return payload


def write_warning_summary(
    ctx: bench.RunContext,
    data_info: Mapping[str, Any],
    task_rows: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    intervention_summary: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(kind: str, severity: str, count: int, detail: str, action: str) -> None:
        rows.append({"warning_kind": kind, "severity": severity, "count": count, "detail": detail, "recommended_action": action})

    if not bool(data_info.get("science_ready")):
        add("not_science_ready", "high", 1, "Run used smoke/fallback data or too few rows for the configured science gate.", "Treat as plumbing only; rerun Tier B with --prompt-set full before ledger claims.")
    eval_rows = [r for r in task_rows if str(r.get("split")) == "eval"]
    if not eval_rows:
        add("missing_eval_rows", "high", 1, "No eval rows survived selection; train/eval language would be unsupported.", "Use a larger or balanced prompt set.")
    if len(task_rows) < SCIENCE_READY_MIN_ROWS:
        add("small_sample_count", "medium", len(task_rows), f"Selected {len(task_rows)} rows; science gate is {SCIENCE_READY_MIN_ROWS}.", "Use Tier B/full for the claim path.")
    unsupported = [r for r in evidence_rows if "supported" not in str(r.get("claim_posture", "")) and "trace_validated" not in str(r.get("claim_posture", ""))]
    if unsupported:
        add("unsupported_evidence_rows", "medium", len(unsupported), "; ".join(str(r.get("method")) for r in unsupported), "Use negative or refinement language for these rows.")
    mismatches = [r for r in trace_rows if not boolish(r.get("result_matches_expected")) or not boolish(r.get("argument_valid"))]
    if mismatches:
        add("trace_or_argument_mismatch", "high", len(mismatches), "At least one deterministic tool trace failed validation.", "Fix the data/tool simulator before interpreting plots.")
    target_shift = summary_value(intervention_summary, "eval", "target_tool_direction", CLAIMABLE_SCALE, "mean_shift_from_zero")
    if not math.isfinite(target_shift):
        target_shift = summary_value(intervention_summary, "all", "target_tool_direction", CLAIMABLE_SCALE, "mean_shift_from_zero")
    random_shift = summary_value(intervention_summary, "eval", "random_direction_control", CLAIMABLE_SCALE, "mean_shift_from_zero")
    if not math.isfinite(random_shift):
        random_shift = summary_value(intervention_summary, "all", "random_direction_control", CLAIMABLE_SCALE, "mean_shift_from_zero")
    if math.isfinite(target_shift) and math.isfinite(random_shift) and target_shift <= random_shift:
        add("random_direction_matches_target", "medium", 1, f"target shift {rounded(target_shift)} <= random shift {rounded(random_shift)}", "Do not write a causal direction claim; inspect dose_response.png and paired_examples.png.")
    if counterexamples:
        add("counterexamples_present", "medium", len(counterexamples), "Automatic counterexamples crossed filters.", "Read tables/failure_specimens.md before using the dashboard.")
    if not rows:
        add("no_runtime_warnings", "info", 0, "No warning conditions crossed the configured filters.", "Still inspect controls and source tables before ledgering claims.")

    csv_path = ctx.path("diagnostics", "warning_summary.csv")
    bench.write_csv_with_context(ctx, csv_path, rows)
    ctx.register_artifact(csv_path, "diagnostic", "Human-readable warning summary for plot and data-quality review.")
    json_path = ctx.path("diagnostics", "warning_summary.json")
    bench.write_json(json_path, {"warnings": rows})
    ctx.register_artifact(json_path, "diagnostic", "JSON warning summary for Lab 34.")
    return rows


def write_failure_specimens(
    ctx: bench.RunContext,
    counterexamples: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
    intervention_rows: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_task = {str(r.get("task_id")): r for r in task_rows}
    by_trace = {str(r.get("task_id")): r for r in trace_rows}
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(counterexamples[:40], start=1):
        task_id = str(row.get("task_id"))
        task = by_task.get(task_id, {})
        trace = by_trace.get(task_id, {})
        target = next((r for r in intervention_rows if str(r.get("task_id")) == task_id and str(r.get("condition")) == "target_tool_direction" and abs(as_float(r.get("scale")) - CLAIMABLE_SCALE) < 1e-9), {})
        random = next((r for r in intervention_rows if str(r.get("task_id")) == task_id and str(r.get("condition")) == "random_direction_control" and abs(as_float(r.get("scale")) - CLAIMABLE_SCALE) < 1e-9), {})
        rows.append({
            "failure_id": f"failure_{i:03d}",
            "kind": row.get("kind"),
            "severity": row.get("severity"),
            "task_id": task_id,
            "split": row.get("split") or task.get("split"),
            "required_tool": row.get("required_tool") or task.get("required_tool"),
            "probe_prediction": row.get("probe_prediction") or task.get("tool_probe_prediction"),
            "surface_prediction": row.get("surface_prediction") or task.get("surface_cue_prediction"),
            "target_shift_at_scale_1": target.get("shift_from_condition_zero", ""),
            "random_shift_at_scale_1": random.get("shift_from_condition_zero", ""),
            "tool_result": trace.get("tool_result", ""),
            "result_matches_expected": trace.get("result_matches_expected", ""),
            "user_prompt": row.get("user_prompt") or task.get("user_prompt", ""),
            "why_it_matters": row.get("why_it_matters", ""),
        })
    jsonl_path = ctx.path("tables", "failure_specimens.jsonl")
    write_jsonl(jsonl_path, rows)
    ctx.register_artifact(jsonl_path, "table", "Counterexample/failure specimens in inspectable JSONL form.")
    md_lines = [
        "# Lab 34 failure specimens",
        "",
        "These are the rows that most shrink the favorite tool-state story. They are evidence, not clutter.",
        "",
    ]
    if rows:
        for row in rows[:16]:
            md_lines += [
                f"## {row['failure_id']}: `{row['kind']}` on `{row['task_id']}`",
                "",
                f"- required tool: `{row.get('required_tool')}`",
                f"- probe prediction: `{row.get('probe_prediction')}`",
                f"- surface prediction: `{row.get('surface_prediction')}`",
                f"- target shift at scale 1: `{row.get('target_shift_at_scale_1')}`",
                f"- random shift at scale 1: `{row.get('random_shift_at_scale_1')}`",
                f"- why it matters: {row.get('why_it_matters')}",
                "",
                "```text",
                str(row.get("user_prompt", ""))[:800],
                "```",
                "",
            ]
    else:
        md_lines.append("No automatic failure specimens crossed the configured filters. This does not prove the claim; it only means the built-in tripwires did not fire.")
    md_path = ctx.path("tables", "failure_specimens.md")
    bench.write_text(md_path, "\n".join(md_lines))
    ctx.register_artifact(md_path, "table", "Markdown failure-specimen cards for quick inspection.")
    return rows


def write_plot_source_tables_and_manifest(
    ctx: bench.RunContext,
    task_rows: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    surface_rows: Sequence[Mapping[str, Any]],
    confusion: Sequence[Mapping[str, Any]],
    intervention_rows: Sequence[Mapping[str, Any]],
    intervention_summary: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    report_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
    arg_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Write figure source tables and a manifest even when --no-plots is used."""
    manifest: list[dict[str, Any]] = []

    def add_manifest(name: str, question: str, source_table: str, row_count: int, metric: str, control: str, claim: str, caveat: str) -> None:
        manifest.append({
            "figure_path": f"plots/{name}" if name.endswith(".png") else name,
            "question_answered": question,
            "source_table": source_table,
            "row_count": row_count,
            "metric": metric,
            "control": control,
            "claim_supported": claim,
            "caveat": caveat,
            "created_when_no_plots": bool(getattr(ctx.args, "no_plots", False) and name.endswith(".png")),
        })

    overview_source = write_figure_source(ctx, "overview_dashboard_source.csv", evidence_rows, "Evidence rows used by the Lab 34 overview/dashboard plot.")
    add_manifest("overview_dashboard.png", "Did decode, control, causal, trace, and failure evidence point in the same direction?", overview_source, len(evidence_rows), "evidence metric values", "named controls in evidence matrix", "Only rows marked supported in the source table.", "Read the source table before using any dashboard cell.")
    add_manifest("tool_use_evidence_dashboard.png", "Backward-compatible main dashboard for Lab 34.", overview_source, len(evidence_rows), "evidence metric values", "named controls in evidence matrix", "Same claim boundary as overview_dashboard.png.", "This copy exists so older handouts still point at a real plot.")

    tvc_rows = source_target_vs_control(task_rows, intervention_rows)
    tvc_source = write_figure_source(ctx, "target_vs_control_source.csv", tvc_rows, "Per-task target-vs-control rows for probe/surface and target/random activation comparisons.")
    add_manifest("target_vs_control.png", "Do target probe and intervention measurements beat their controls?", tvc_source, len(tvc_rows), "paired task-level differences", "surface heuristic and random direction", "A target/control gap supports only the scoped toy-harness signal.", "Raw paired rows matter more than a polished mean.")

    dose_source = write_figure_source(ctx, "dose_response_source.csv", intervention_rows, "Raw constrained action-letter intervention rows by task, condition, and scale.")
    dose_summary_source = write_figure_source(ctx, "dose_response_summary_source.csv", intervention_summary, "Aggregate constrained action-letter intervention rows by split, condition, and scale.")
    add_manifest("dose_response.png", "Does the action-letter effect grow with scale and beat random direction?", dose_source, len(intervention_rows), "shift_from_condition_zero", "random_direction_control", "A narrow causal handle if target direction beats random at claimable scale.", "This remains an A/B/C/D/E/F/N letter-prompt result.")
    add_manifest("tool_state_patch_recovery.png", "Legacy dose-response plot for activation-addition recovery.", dose_summary_source, len(intervention_summary), "mean_shift_from_zero", "random_direction_control", "Same claim boundary as dose_response.png.", "Do not read as open-ended tool reliability.")

    layer_source = write_figure_source(ctx, "layer_sweep_heatmap_source.csv", probe_rows, "Probe metrics by residual depth and split.")
    add_manifest("layer_sweep_heatmap.png", "Where do decode and surface-control metrics sit across depth?", layer_source, len(probe_rows), "tool_needed_auc/tool_selection_accuracy/control gap", "surface and shuffled controls", "Depth evidence is candidate support only after train/eval discipline.", "A bright layer is not a circuit.")
    add_manifest("tool_choice_probe_by_depth.png", "How do train/eval decode metrics change across residual depth?", layer_source, len(probe_rows), "AUC and accuracy by depth", "surface and shuffled controls", "Shows depth context for the selected site.", "Do not select from eval brightness.")

    traj_source = write_figure_source(ctx, "trajectory_source.csv", transition_rows, "Tool-state transition rows used by the trajectory plot.")
    add_manifest("trajectory.png", "What deterministic toy-tool trace did the harness execute?", traj_source, len(transition_rows), "step/tool counts", "closed local simulator", "Trace audit only, not self-report.", "The model did not generate this trace.")

    confusion_source = write_figure_source(ctx, "tool_selection_confusion_matrix_source.csv", confusion, "Required-tool versus predicted-tool counts.")
    add_manifest("tool_selection_confusion_matrix.png", "Which tools are confused with which other tools?", confusion_source, len(confusion), "count", "required-vs-predicted matrix", "Confusions bound which-tool language.", "No-tool false positives matter more than pretty diagonals.")

    selected_depths = {int(r.get("selected_depth", 0)) for r in task_rows if str(r.get("selected_depth", "")).strip()}
    selected_depth = next(iter(selected_depths), 0)
    selected = selected_probe_row(probe_rows, selected_depth, "eval")
    surface_source_rows = [
        {"metric": "residual_probe_eval_accuracy", "value": safe_mean([1.0 if boolish(r.get("tool_probe_correct")) else 0.0 for r in task_rows if r.get("split") == "eval"], 0.0), "source": "task_manifest"},
        {"metric": "surface_heuristic_eval_accuracy", "value": safe_mean([1.0 if boolish(r.get("surface_cue_correct")) else 0.0 for r in task_rows if r.get("split") == "eval"], 0.0), "source": "task_manifest"},
        {"metric": "shuffled_label_control_accuracy", "value": as_float(selected.get("shuffled_label_control_accuracy"), 0.0) if selected else 0.0, "source": "tool_choice_probe_report"},
        {"metric": "no_tool_false_positive_rate", "value": as_float(selected.get("no_tool_false_positive_rate"), 0.0) if selected else 0.0, "source": "tool_choice_probe_report"},
    ]
    surface_source = write_figure_source(ctx, "surface_control_ladder_source.csv", surface_source_rows, "Source values for surface/shuffled/no-tool control ladder.")
    add_manifest("surface_control_ladder.png", "Does residual decoding beat the boring lexical heuristic?", surface_source, len(surface_source_rows), "accuracy/rate", "surface and shuffled controls", "Supported only when residual probe beats surface on eval.", "If surface wins, the negative result is the lesson.")

    mem_rows: list[dict[str, Any]] = []
    read_counts = defaultdict(int)
    for row in trace_rows:
        fam = str(row.get("family"))
        reads = safe_json_list(row.get("memory_reads_json"))
        read_counts[fam] += len(reads)
        mem_rows.append({"task_id": row.get("task_id"), "family": fam, "n_memory_reads": len(reads), "memory_reads_json": row.get("memory_reads_json")})
    mem_source = write_figure_source(ctx, "memory_read_trace_atlas_source.csv", mem_rows, "Memory-read counts derived from deterministic trace rows.")
    add_manifest("memory_read_trace_atlas.png", "Which toy tools read synthetic memory objects?", mem_source, len(mem_rows), "memory read count", "closed local simulator", "Only a harness trace claim.", "Do not infer model memory reads.")

    reliance_rows: list[dict[str, Any]] = []
    for row in trace_rows:
        val = 1.0 if boolish(row.get("would_final_answer_change_if_tool_result_corrupted")) else 0.0
        reliance_rows.append({"task_id": row.get("task_id"), "family": row.get("family"), "would_change": val})
    reliance_source = write_figure_source(ctx, "tool_result_reliance_ladder_source.csv", reliance_rows, "Source rows for corrupted-result reliance by family.")
    add_manifest("tool_result_reliance_ladder.png", "Would corrupting the toy tool result change the toy final answer?", reliance_source, len(reliance_rows), "fraction would change", "corrupted tool result", "Only a deterministic-trace reliance claim.", "Does not show the model verifies tool outputs.")

    self_report_source = write_figure_source(ctx, "tool_self_report_matrix_source.csv", report_rows, "Known-trace self-report review scaffold rows.")
    add_manifest("tool_self_report_matrix.png", "Which rows require human review before self-report/source-attribution language?", self_report_source, len(report_rows), "review/match counts", "blank review columns", "Review scaffold only.", "No model self-report is generated by the default run.")

    failure_source = write_figure_source(ctx, "paired_examples_source.csv", counterexamples, "Counterexample rows used for the paired_examples plot.")
    add_manifest("paired_examples.png", "Which concrete rows most weaken the favorite claim?", failure_source, len(counterexamples), "severity", "counterexample filters", "Counterexamples define claim boundaries.", "No automatic failures is not proof of no failures.")

    write_figure_source(ctx, "surface_cue_audit_source.csv", surface_rows, "Surface-cue audit rows mirrored for figure provenance.")
    write_figure_source(ctx, "tool_depth_selection_source.csv", selection_rows, "Train-side depth-selection rows mirrored for figure provenance.")
    write_figure_source(ctx, "tool_argument_validation_source.csv", arg_rows, "Tool argument validation rows mirrored for figure provenance.")
    write_plot_manifest(ctx, manifest)
    return manifest

# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def write_safety_status(ctx: bench.RunContext, data_info: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "lab": LAB_ID,
        "safe_scope": data_info["safety_scope"],
        "allowed_tools": ["calculator", "dictionary", "calendar", "file_search", "route_planner", "unit_converter", "none"],
        "blocked_activities": [
            "web browsing",
            "real credentials",
            "real filesystem reads or writes",
            "real calendar access",
            "network calls",
            "harmful tools",
            "autonomous deployment",
        ],
        "tool_implementation": "deterministic local simulators over synthetic data",
        "science_ready": data_info.get("science_ready"),
    }
    path = ctx.path("diagnostics", "safety_status.json")
    bench.write_json(path, payload)
    ctx.register_artifact(path, "diagnostic", "Safety and scope status for Lab 34.")
    return payload


def write_self_check_status(
    ctx: bench.RunContext,
    token_rows: Sequence[Mapping[str, Any]],
    arg_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    status = {
        "tokenization_kept": sum(1 for r in token_rows if r.get("kept")),
        "tokenization_dropped": sum(1 for r in token_rows if not r.get("kept")),
        "argument_valid_rate": rounded(safe_mean([1.0 if r.get("argument_valid") else 0.0 for r in arg_rows], 0.0)),
        "trace_result_match_rate": rounded(safe_mean([1.0 if r.get("result_matches_expected") else 0.0 for r in arg_rows], 0.0)),
        "evidence_rows": len(evidence_rows),
    }
    status["ok"] = bool(status["tokenization_kept"] > 0 and as_float(status["argument_valid_rate"]) >= 0.999 and as_float(status["trace_result_match_rate"]) >= 0.999)
    path = ctx.path("diagnostics", "self_check_status.json")
    bench.write_json(path, status)
    ctx.register_artifact(path, "diagnostic", "Lab 34 tokenization, tool-argument, and evidence self-check status.")
    return status


def write_tables(
    ctx: bench.RunContext,
    task_rows: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    surface_rows: Sequence[Mapping[str, Any]],
    confusion: Sequence[Mapping[str, Any]],
    intervention_rows: Sequence[Mapping[str, Any]],
    intervention_summary: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    report_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
    arg_rows: Sequence[Mapping[str, Any]],
) -> None:
    specs = [
        ("results.csv", task_rows, "Task-level selected-depth probe predictions and review fields."),
        ("tables/tool_task_manifest.csv", task_rows, "Task-level tool probe predictions, surface controls, features, and review fields."),
        ("tables/tool_choice_probe_report.csv", probe_rows, "Tool-needed and tool-selection probe metrics by depth and split."),
        ("tables/tool_depth_selection.csv", selection_rows, "Train-side depth-selection receipt for Lab 34."),
        ("tables/surface_cue_audit.csv", surface_rows, "Surface-cue heuristic predictions and prompt-feature audit."),
        ("tables/tool_confusion_matrix.csv", confusion, "Required-tool versus selected-depth probe prediction counts."),
        ("tables/tool_intervention_report.csv", intervention_rows, "Activation-addition rows for the constrained action-letter prompt."),
        ("tables/tool_intervention_summary.csv", intervention_summary, "Aggregate action-letter intervention shifts by split, condition, and scale."),
        ("tables/tool_trace_log.csv", trace_rows, "Deterministic toy-tool trace log."),
        ("tables/tool_state_transition_log.csv", transition_rows, "Per-task state-transition trace from prompt to local tool result."),
        ("tables/tool_self_report_labels.csv", report_rows, "Known-trace self-report review scaffold with blank human-label columns."),
        ("tables/tool_use_evidence_matrix.csv", evidence_rows, "Lab 34 evidence matrix and claim posture."),
        ("tables/evidence_matrix.csv", evidence_rows, "Standard evidence-matrix alias for Lab 34 claim posture."),
        ("tables/tool_counterexamples.csv", counterexamples, "Rows that weaken or falsify the favorite tool-state claim."),
        ("diagnostics/tool_argument_validation.csv", arg_rows, "Tool argument validation and deterministic result checks."),
    ]
    for rel, rows, desc in specs:
        path = ctx.path(*rel.split("/"))
        bench.write_csv_with_context(ctx, path, rows)
        ctx.register_artifact(path, "table" if not rel.startswith("diagnostics") else "diagnostic", desc)

    # JSONL mirrors keep the most important row-level artifacts inspectable by
    # streaming tools and make it easier to diff smoke and science runs.
    jsonl_specs = [
        ("results.jsonl", task_rows, "JSONL mirror of selected-depth task rows."),
        ("tables/tool_task_manifest.jsonl", task_rows, "JSONL mirror of the task manifest."),
        ("tables/tool_intervention_report.jsonl", intervention_rows, "JSONL mirror of action-letter intervention rows."),
        ("tables/tool_trace_log.jsonl", trace_rows, "JSONL mirror of deterministic tool trace rows."),
    ]
    for rel, rows, desc in jsonl_specs:
        path = ctx.path(*rel.split("/"))
        write_jsonl(path, [{**ctx.table_context(), **dict(row)} for row in rows])
        ctx.register_artifact(path, "table", desc)


def write_state(ctx: bench.RunContext, model: DirectionModel, all_metadata: Mapping[int, Mapping[str, Any]]) -> None:
    import torch

    path = ctx.path("state", "tool_directions.pt")
    torch.save({
        "selected_depth": model.depth,
        "steer_layer": model.steer_layer,
        "unit": {name: vec.cpu() for name, vec in model.unit.items()},
        "steer": {name: vec.cpu() for name, vec in model.steer.items()},
    }, path)
    ctx.register_artifact(path, "state", "Selected Lab 34 tool-needed, tool-selection, and control directions.")
    meta_path = ctx.path("state", "tool_direction_metadata.json")
    bench.write_json(meta_path, {"selected": model.metadata, "all_depths": all_metadata})
    ctx.register_artifact(meta_path, "state", "Tool direction depth, threshold, and norm metadata.")


def write_plot_reading_guide(ctx: bench.RunContext) -> None:
    rows = [
        {"plot": "plots/overview_dashboard.png", "read_for": "The shortest overview: decode, control, causal, trace, and failure load.", "source_table": "tables/figure_sources/overview_dashboard_source.csv", "do_not_claim": "A dashboard pass is not evidence for autonomous planning."},
        {"plot": "plots/tool_use_evidence_dashboard.png", "read_for": "Backward-compatible dashboard name used as the main plot in the original handout.", "source_table": "tables/figure_sources/overview_dashboard_source.csv", "do_not_claim": "Dashboard aesthetics cannot upgrade weak controls."},
        {"plot": "plots/target_vs_control.png", "read_for": "Per-task residual probe versus surface baseline, and target-direction versus random-direction causal shifts.", "source_table": "tables/figure_sources/target_vs_control_source.csv", "do_not_claim": "One or two wins imply real-world tool reliability."},
        {"plot": "plots/dose_response.png", "read_for": "Whether action-letter shifts grow with dose and whether target directions separate from random controls.", "source_table": "tables/figure_sources/dose_response_source.csv", "do_not_claim": "A monotonic letter-prompt shift is open-ended competence."},
        {"plot": "plots/layer_sweep_heatmap.png", "read_for": "Selected-depth context across decode, surface, and gap metrics.", "source_table": "tables/figure_sources/layer_sweep_heatmap_source.csv", "do_not_claim": "A bright layer is a full circuit."},
        {"plot": "plots/trajectory.png", "read_for": "The deterministic toy trace path from prompt to oracle tool to result to final answer.", "source_table": "tables/figure_sources/trajectory_source.csv", "do_not_claim": "Harness trace equals model introspection."},
        {"plot": "plots/paired_examples.png", "read_for": "Counterexamples and failure specimens sorted by severity.", "source_table": "tables/figure_sources/paired_examples_source.csv", "do_not_claim": "No automatic specimens means no failures exist."},
        {"plot": "plots/tool_choice_probe_by_depth.png", "read_for": "Train/eval probe scores by residual depth with surface controls nearby.", "source_table": "tables/figure_sources/tool_choice_probe_by_depth_source.csv", "do_not_claim": "Depth trends identify a full tool-use circuit."},
        {"plot": "plots/tool_selection_confusion_matrix.png", "read_for": "Required tool versus predicted tool at the selected depth.", "source_table": "tables/figure_sources/tool_selection_confusion_matrix_source.csv", "do_not_claim": "Toy-tool accuracy transfers to real agents."},
        {"plot": "plots/tool_state_patch_recovery.png", "read_for": "Legacy name for the activation-addition dose-response curve.", "source_table": "tables/figure_sources/dose_response_source.csv", "do_not_claim": "Letter-prompt shifts are open-ended tool competence."},
        {"plot": "plots/surface_control_ladder.png", "read_for": "Residual probe versus surface and shuffled controls.", "source_table": "tables/figure_sources/surface_control_ladder_source.csv", "do_not_claim": "Probe accuracy is meaningful if surface cues match it."},
        {"plot": "plots/memory_read_trace_atlas.png", "read_for": "Known trace reads by family.", "source_table": "tables/figure_sources/memory_read_trace_atlas_source.csv", "do_not_claim": "Harness trace is a cognitive map."},
        {"plot": "plots/tool_result_reliance_ladder.png", "read_for": "Would corrupted tool results alter the toy final answer.", "source_table": "tables/figure_sources/tool_result_reliance_ladder_source.csv", "do_not_claim": "The model would verify corrupted results."},
        {"plot": "plots/tool_self_report_matrix.png", "read_for": "Known-trace label scaffold and human-review requirement.", "source_table": "tables/figure_sources/tool_self_report_matrix_source.csv", "do_not_claim": "The model knows why it used the tool."},
    ]
    path = ctx.path("tables", "plot_reading_guide.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Reading guide for the Lab 34 plot suite, including source tables.")


def write_method_card(ctx: bench.RunContext, data_info: Mapping[str, Any], model: DirectionModel, evidence: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 34 method card",
        "",
        "This lab studies controlled toy-tool traces. It does not claim persistent goals or autonomous plans.",
        "",
        f"- model: `{ctx.model_id}`",
        f"- selected residual depth: `{model.depth}`",
        f"- steering layer for activation addition: `{model.steer_layer}`",
        f"- data source: `{data_info.get('data_source')}`",
        f"- science-ready: `{data_info.get('science_ready')}`",
        "- tools: calculator, dictionary, calendar, file_search, route_planner, unit_converter, none",
        "- decode object: final-token prompt-boundary residual state",
        "- causal object: constrained A/B/C/D/E/F/N action-letter prompt",
        "- main null: surface cues explain tool choice",
        "- self-report labels: known-trace review templates, not model introspection",
        "- forbidden claim: the model has a persistent goal or autonomous plan",
        "",
        "## Headline metrics",
        "",
        f"- tool-needed eval AUC: `{metrics.get('tool_needed_auc')}`",
        f"- tool-selection eval accuracy: `{metrics.get('tool_selection_accuracy')}`",
        f"- surface-control eval accuracy: `{metrics.get('surface_control_accuracy')}`",
        f"- causal shift over random: `{metrics.get('causal_shift_over_random')}`",
        f"- counterexamples: `{metrics.get('n_counterexamples')}`",
        "",
        "## Evidence rows",
        "",
        "| method | rung | value | control | posture |",
        "|---|---|---:|---:|---|",
    ]
    for row in evidence:
        lines.append(f"| `{row['method']}` | {row['evidence_rung']} | {row['value']} | {row['control_value']} | `{row['claim_posture']}` |")
    lines += [
        "",
        "## Claim boundary",
        "",
        "A positive row supports a toy-harness signal claim only. It does not support intention, planning, real-world tool reliability, or faithful self-report.",
        "",
    ]
    path = ctx.path("method_card.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Lab 34 method card and non-claim boundary.")


def write_operationalization_audit(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]], counterexamples: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Lab 34 operationalization audit",
        "",
        "```yaml",
        "headline_claim: \"a prompt-boundary state tracks toy-tool need and tool choice\"",
        "cheap_explanation: \"digits, tool names, filenames, route words, units, or answer scaffolding explain the result\"",
        "killer_control: \"surface-cue heuristic, no-tool cue rows, shuffled labels, random direction, corrupted-result trace, and human review columns\"",
        "result: \"filled by the evidence matrix below\"",
        "claim_allowed: \"toy-harness handle, not autonomous planning\"",
        "```",
        "",
        "## Cheap explanations and controls",
        "",
        "| Cheap explanation | Control | What would make the cheap explanation win? |",
        "|---|---|---|",
        "| Digits imply calculator | surface heuristic and no-tool digit rows | surface accuracy matches or beats the probe |",
        "| Tool names imply tool choice | no-tool rows with tool words | probe false positives on `required_tool=none` |",
        "| Lookup words blur dictionary and file search | confusion matrix | dictionary/file-search confusion dominates eval |",
        "| Action-letter prompt has priors | random direction control | random shift matches target direction shift |",
        "| Harness trace is mistaken for self-report | review scaffold | claims cite `tool_self_report_labels.csv` before review columns are filled |",
        "| Tool result reliance is overread | corrupted-result flag | writeup claims verification or robustness that was not measured |",
        "",
        "## Verdicts",
        "",
    ]
    for row in evidence:
        lines.append(f"- `{row['method']}`: `{row['claim_posture']}` with value `{row['value']}` and control `{row['control_value']}`.")
    lines += ["", "## Counterexamples", ""]
    if counterexamples:
        for row in counterexamples[:12]:
            lines.append(f"- `{row['kind']}` on `{row['task_id']}`: {row.get('why_it_matters', '')}")
    else:
        lines.append("- No automatic counterexamples crossed the configured filters. Replicate before broadening the claim.")
    lines += [
        "",
        "## Allowed language",
        "",
        "- `On this toy harness, the selected prompt-boundary state predicted tool labels above named controls.`",
        "- `Activation addition shifted constrained action-letter logits above random controls.`",
        "",
        "## Forbidden language",
        "",
        "- `The model has a persistent goal or autonomous plan.`",
        "- `The tool direction is intention.`",
        "- `The model knows why it used the tool.`",
        "- `The toy harness proves real-world agent reliability.`",
        "",
    ]
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Lab 34 controls, counterexamples, and allowed claim grammar.")


def write_run_summary(ctx: bench.RunContext, data_info: Mapping[str, Any], metrics: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Lab 34 run summary: tool use, agents, and state tracking",
        "",
        "## Run identity",
        "",
        f"- model: `{ctx.model_id}`",
        f"- data: `{pathlib.Path(str(data_info['data_path'])).name}` sha256 `{str(data_info['sha256'])[:16]}`",
        f"- selected rows: {data_info['n_rows_selected']} from {data_info['n_rows_file']}",
        f"- required tools: `{data_info['required_tools']}`",
        f"- splits: `{data_info['splits']}`",
        f"- science-ready: `{data_info['science_ready']}`",
        "- intervention: activation addition on a constrained action-letter prompt",
        "- evidence: `OBS + DECODE + CAUSAL + SELF-REPORT`, scoped to toy tools",
        "",
        "## Headline verdicts",
        "",
        "| method | rung | metric | value | control | posture |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in evidence:
        lines.append(f"| `{row['method']}` | {row['evidence_rung']} | {row['metric']} | {row['value']} | {row['control_value']} | `{row['claim_posture']}` |")
    lines += [
        "",
        "## Reading order",
        "",
        "1. `method_card.md` for the claim boundary.",
        "2. `diagnostics/safety_status.json` and `diagnostics/self_check_status.json` for guardrails.",
        "3. `tables/surface_cue_audit.csv` before any decode claim.",
        "4. `tables/tool_depth_selection.csv` and `tables/tool_choice_probe_report.csv` for split-aware decode.",
        "5. `tables/tool_task_manifest.csv` and `tables/tool_confusion_matrix.csv` for row-level failures.",
        "6. `tables/tool_intervention_summary.csv` for the causal letter-prompt test.",
        "7. `tables/tool_trace_log.csv` and `tables/tool_self_report_labels.csv` before source-attribution language.",
        "8. `tables/tool_counterexamples.csv` before writing a positive claim.",
        "",
        "## Smallest surviving claim",
        "",
        "The run can support only the rows marked supported in `tables/tool_use_evidence_matrix.csv`, and only for this toy harness, dataset, model, selected depth, and controls.",
        "",
        "## Non-claims",
        "",
        "This run does not show persistent goals, autonomous planning, real-world agent reliability, or faithful introspection.",
        "",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Lab 34 run summary and reading order.")


def write_ledger_claims(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]]) -> None:
    run_name = ctx.run_dir.name
    claims = []
    for i, row in enumerate(evidence, start=1):
        posture = str(row["claim_posture"])
        if "supported" in posture or posture == "trace_validated":
            text = (
                f"Lab 34 method `{row['method']}` reported {row['metric']}={row['value']} against control "
                f"{row['control_metric']}={row['control_value']} with posture `{posture}`. This is a toy-harness signal claim, not an autonomous-plan claim."
            )
        else:
            text = (
                f"Lab 34 method `{row['method']}` did not earn broad positive language: {row['metric']}={row['value']} "
                f"against control {row['control_metric']}={row['control_value']} produced posture `{posture}`."
            )
        claims.append({
            "id": f"{LAB_ID}-C{i}",
            "tag": str(row["evidence_rung"]),
            "text": text,
            "artifact": f"runs/{run_name}/tables/tool_use_evidence_matrix.csv",
            "falsifier": "A held-out surface-cue no-tool set, shuffled labels, or random activation direction matches or beats the measured tool signal.",
        })
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def write_plots(
    ctx: bench.RunContext,
    task_rows: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    intervention_rows: Sequence[Mapping[str, Any]],
    intervention_summary: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    report_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
) -> None:
    write_plot_reading_guide(ctx)
    if ctx.args.no_plots:
        return
    import matplotlib.pyplot as plt
    import numpy as np

    manifest: list[dict[str, Any]] = []

    def add_manifest(name: str, question: str, source_table: str, row_count: int, metric: str, control: str, claim: str, caveat: str) -> None:
        manifest.append({
            "figure_path": f"plots/{name}",
            "question_answered": question,
            "source_table": source_table,
            "row_count": row_count,
            "metric": metric,
            "control": control,
            "claim_supported": claim,
            "caveat": caveat,
        })

    # ------------------------------------------------------------------
    # Overview dashboard, also saved to the original dashboard filename.
    # ------------------------------------------------------------------
    overview_source = write_figure_source(ctx, "overview_dashboard_source.csv", evidence_rows, "Evidence rows used by the Lab 34 overview/dashboard plot.")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    fig.suptitle("Lab 34 tool-use evidence dashboard", fontsize=14, fontweight="bold")
    names = [str(r["method"]).replace("prompt_boundary_", "").replace("constrained_", "") for r in evidence_rows]
    vals = [as_float(r.get("value"), 0.0) for r in evidence_rows]
    ctrls = [as_float(r.get("control_value"), 0.0) for r in evidence_rows]
    x = np.arange(len(names))
    if names:
        axes[0, 0].bar(x - 0.18, vals, 0.36, label="measured")
        axes[0, 0].bar(x + 0.18, ctrls, 0.36, label="control")
        axes[0, 0].set_xticks(x, names, rotation=35, ha="right", fontsize=7)
        axes[0, 0].legend(fontsize=8)
    else:
        add_empty_message(axes[0, 0], "No evidence rows")
    axes[0, 0].set_title("Evidence values beside controls")

    eval_needed = [r for r in probe_rows if r.get("split_group") == "eval"] or [r for r in probe_rows if r.get("split_group") == "all"]
    if eval_needed:
        depths = [int(r["depth"]) for r in eval_needed]
        axes[0, 1].plot(depths, [as_float(r.get("tool_needed_auc"), 0.0) for r in eval_needed], marker="o", label="needed AUC")
        axes[0, 1].plot(depths, [as_float(r.get("tool_selection_accuracy"), 0.0) for r in eval_needed], marker="s", label="tool acc")
        axes[0, 1].plot(depths, [as_float(r.get("surface_control_accuracy"), 0.0) for r in eval_needed], marker="^", label="surface acc")
        axes[0, 1].set_ylim(0, 1.05)
        axes[0, 1].legend(fontsize=8)
    else:
        add_empty_message(axes[0, 1], "No probe rows")
    axes[0, 1].set_title("Decode by depth")
    axes[0, 1].set_xlabel("residual stream depth")

    scale_rows = [r for r in intervention_summary if r.get("split_group") in {"eval", "all"} and abs(as_float(r.get("scale")) - CLAIMABLE_SCALE) < 1e-9]
    conds = sorted({str(r["condition"]) for r in scale_rows})
    if conds:
        axes[1, 0].bar(conds, [safe_mean([r.get("mean_shift_from_zero") for r in scale_rows if r.get("condition") == c], 0.0) for c in conds])
        axes[1, 0].set_xticks(range(len(conds)), conds, rotation=30, ha="right")
        axes[1, 0].axhline(0, linewidth=0.8)
    else:
        add_empty_message(axes[1, 0], "No scale-1 intervention rows")
    axes[1, 0].set_title("Scale-1 action-letter shift")

    counts = Counter(str(r.get("kind")) for r in counterexamples)
    c_labels = sorted(counts) or ["none"]
    c_vals = [counts[k] for k in c_labels] if counts else [0]
    axes[1, 1].bar(c_labels, c_vals)
    axes[1, 1].set_xticks(range(len(c_labels)), c_labels, rotation=25, ha="right", fontsize=8)
    axes[1, 1].set_title("Automatic counterexamples")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    bench.save_figure(ctx, fig, "overview_dashboard.png", "Lab 34 overview dashboard with decode, controls, causal shift, and failure load.")
    add_manifest("overview_dashboard.png", "Did decode, control, causal, trace, and failure evidence point in the same direction?", overview_source, len(evidence_rows), "evidence metric values", "named controls in evidence matrix", "Only the rows marked supported in the source table.", "Read the source table before using any dashboard cell.")

    # Backward-compatible dashboard filename expected by the original handout.
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    fig.suptitle("Lab 34 tool-use evidence dashboard", fontsize=14, fontweight="bold")
    if names:
        axes[0, 0].bar(x - 0.18, vals, 0.36, label="measured")
        axes[0, 0].bar(x + 0.18, ctrls, 0.36, label="control")
        axes[0, 0].set_xticks(x, names, rotation=35, ha="right", fontsize=7)
        axes[0, 0].legend(fontsize=8)
    else:
        add_empty_message(axes[0, 0], "No evidence rows")
    axes[0, 0].set_title("Evidence values beside controls")
    if eval_needed:
        depths = [int(r["depth"]) for r in eval_needed]
        axes[0, 1].plot(depths, [as_float(r.get("tool_needed_auc"), 0.0) for r in eval_needed], marker="o", label="needed AUC")
        axes[0, 1].plot(depths, [as_float(r.get("tool_selection_accuracy"), 0.0) for r in eval_needed], marker="s", label="tool acc")
        axes[0, 1].plot(depths, [as_float(r.get("surface_control_accuracy"), 0.0) for r in eval_needed], marker="^", label="surface acc")
        axes[0, 1].set_ylim(0, 1.05)
        axes[0, 1].legend(fontsize=8)
    else:
        add_empty_message(axes[0, 1], "No probe rows")
    axes[0, 1].set_title("Decode by depth")
    axes[0, 1].set_xlabel("residual stream depth")
    if conds:
        axes[1, 0].bar(conds, [safe_mean([r.get("mean_shift_from_zero") for r in scale_rows if r.get("condition") == c], 0.0) for c in conds])
        axes[1, 0].set_xticks(range(len(conds)), conds, rotation=30, ha="right")
        axes[1, 0].axhline(0, linewidth=0.8)
    else:
        add_empty_message(axes[1, 0], "No scale-1 intervention rows")
    axes[1, 0].set_title("Scale-1 action-letter shift")
    axes[1, 1].bar(c_labels, c_vals)
    axes[1, 1].set_xticks(range(len(c_labels)), c_labels, rotation=25, ha="right", fontsize=8)
    axes[1, 1].set_title("Automatic counterexamples")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    bench.save_figure(ctx, fig, "tool_use_evidence_dashboard.png", "Lab 34 tool-use evidence dashboard, retained under the original filename.")
    add_manifest("tool_use_evidence_dashboard.png", "Backward-compatible main dashboard for Lab 34.", overview_source, len(evidence_rows), "evidence metric values", "named controls in evidence matrix", "Same claim boundary as overview_dashboard.png.", "This copy exists so older handouts still point at a real plot.")

    # ------------------------------------------------------------------
    # Target-vs-control paired raw view.
    # ------------------------------------------------------------------
    tvc_rows = source_target_vs_control(task_rows, intervention_rows)
    tvc_source = write_figure_source(ctx, "target_vs_control_source.csv", tvc_rows, "Per-task target-vs-control rows for probe/surface and target/random activation comparisons.")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    probe_part = [r for r in tvc_rows if r.get("source_part") == "probe_vs_surface"]
    if probe_part:
        probe_part = sorted(probe_part, key=lambda r: (str(r.get("required_tool")), as_float(r.get("difference_target_minus_control")), str(r.get("task_id"))))
        xs = np.arange(len(probe_part))
        for i, row in enumerate(probe_part):
            axes[0].plot([i, i], [as_float(row.get("control_value")), as_float(row.get("target_value"))], linewidth=0.8, alpha=0.65)
        axes[0].scatter(xs, [as_float(r.get("control_value")) for r in probe_part], label="surface")
        axes[0].scatter(xs, [as_float(r.get("target_value")) for r in probe_part], marker="x", label="residual probe")
        axes[0].set_ylim(-0.08, 1.08)
        axes[0].set_ylabel("correct on task")
        axes[0].legend(fontsize=8)
    else:
        add_empty_message(axes[0], "No per-task probe rows")
    axes[0].set_title("Probe vs surface on held-out tasks")
    axes[0].set_xlabel("task, sorted by required tool")

    causal_part = [r for r in tvc_rows if r.get("source_part") == "activation_target_vs_random"]
    if causal_part:
        causal_part = sorted(causal_part, key=lambda r: (str(r.get("required_tool")), as_float(r.get("difference_target_minus_control")), str(r.get("task_id"))))
        xs = np.arange(len(causal_part))
        for i, row in enumerate(causal_part):
            axes[1].plot([i, i], [as_float(row.get("control_value")), as_float(row.get("target_value"))], linewidth=0.8, alpha=0.65)
        axes[1].scatter(xs, [as_float(r.get("control_value")) for r in causal_part], label="random")
        axes[1].scatter(xs, [as_float(r.get("target_value")) for r in causal_part], marker="x", label="target direction")
        axes[1].axhline(0, linewidth=0.8)
        axes[1].set_ylabel("shift from scale 0")
        axes[1].legend(fontsize=8)
    else:
        add_empty_message(axes[1], "No scale-1 target/random pairs")
    axes[1].set_title("Action-letter shift: target vs random")
    axes[1].set_xlabel("task, sorted by required tool")
    fig.suptitle("Target measurements beside their controls", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    bench.save_figure(ctx, fig, "target_vs_control.png", "Per-task target measurements beside surface/random controls.")
    add_manifest("target_vs_control.png", "Do raw task pairs show targets beating controls, or only aggregate glitter?", tvc_source, len(tvc_rows), "probe correctness and action-letter shift", "surface heuristic and random direction", "Supported only if per-task wins are not isolated anecdotes.", "Tiny Tier A rows should be read as a plumbing check.")

    # ------------------------------------------------------------------
    # Dose response, source-backed raw rows.
    # ------------------------------------------------------------------
    dose_source = write_figure_source(ctx, "dose_response_source.csv", intervention_rows, "Raw intervention rows used for dose-response plots.")
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    split = "eval" if any(str(r.get("split_group")) == "eval" for r in intervention_summary) else "all"
    summary_for_split = [r for r in intervention_summary if str(r.get("split_group")) == split]
    for condition in sorted({str(r.get("condition")) for r in summary_for_split}):
        rows = sorted([r for r in summary_for_split if str(r.get("condition")) == condition], key=lambda r: as_float(r.get("scale")))
        ax.plot([as_float(r.get("scale")) for r in rows], [as_float(r.get("mean_shift_from_zero")) for r in rows], marker="o", label=condition)
    raw_for_split = [r for r in intervention_rows if split == "all" or str(r.get("split")) == split]
    # Add faint raw points so a mean line cannot hide ragged rows.
    for condition in sorted({str(r.get("condition")) for r in raw_for_split}):
        rows = [r for r in raw_for_split if str(r.get("condition")) == condition]
        if not rows:
            continue
        ax.scatter([as_float(r.get("scale")) for r in rows], [as_float(r.get("shift_from_condition_zero")) for r in rows], s=14, alpha=0.35)
    if not summary_for_split:
        add_empty_message(ax, "No intervention rows")
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("intervention scale")
    ax.set_ylabel("target-letter margin shift from scale 0")
    ax.set_title(f"Tool-choice activation-addition dose response ({split})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "dose_response.png", "Dose-response curve for target, needed, and random directions with raw points.")
    add_manifest("dose_response.png", "Does target-direction shift separate from random across intervention strength?", dose_source, len(intervention_rows), "target-minus-distractor logit shift", "random_direction_control", "Narrow causal claim only if target shift beats random at the predeclared scale.", "This is an A/B/C/D/E/F/N prompt, not open tool use.")

    # Legacy dose plot filename.
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for condition in sorted({str(r.get("condition")) for r in summary_for_split}):
        rows = sorted([r for r in summary_for_split if str(r.get("condition")) == condition], key=lambda r: as_float(r.get("scale")))
        ax.plot([as_float(r.get("scale")) for r in rows], [as_float(r.get("mean_shift_from_zero")) for r in rows], marker="o", label=condition)
    if not summary_for_split:
        add_empty_message(ax, "No intervention rows")
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("intervention scale")
    ax.set_ylabel("mean shift from scale 0")
    ax.set_title(f"Tool-state activation-addition recovery ({split})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "tool_state_patch_recovery.png", "Legacy dose-response plot for action-letter activation addition.")
    add_manifest("tool_state_patch_recovery.png", "Legacy view of the activation-addition dose response.", dose_source, len(intervention_rows), "mean shift from zero", "random_direction_control", "Same as dose_response.png.", "Prefer dose_response.png for raw-point overlays.")

    # ------------------------------------------------------------------
    # Layer/depth sweep heatmap.
    # ------------------------------------------------------------------
    heat_source = write_figure_source(ctx, "layer_sweep_heatmap_source.csv", probe_rows, "Probe metrics by residual depth and split for the layer-sweep heatmap.")
    split = "eval" if any(str(r.get("split_group")) == "eval" for r in probe_rows) else "all"
    heat_rows = sorted([r for r in probe_rows if str(r.get("split_group")) == split], key=lambda r: int(r.get("depth", 0)))
    metrics = [
        ("tool_needed_auc", "needed AUC"),
        ("tool_needed_accuracy", "needed acc"),
        ("tool_selection_accuracy", "tool acc"),
        ("surface_control_accuracy", "surface acc"),
        ("decode_gap_over_surface", "probe-surface"),
        ("no_tool_false_positive_rate", "no-tool FP"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    if heat_rows:
        mat = np.array([[as_float(r.get(key), 0.0) for r in heat_rows] for key, _label in metrics], dtype=float)
        im = ax.imshow(mat, aspect="auto")
        ax.set_xticks(range(len(heat_rows)), [str(r.get("depth")) for r in heat_rows])
        ax.set_yticks(range(len(metrics)), [label for _key, label in metrics])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
    else:
        add_empty_message(ax, "No probe depth rows")
    ax.set_xlabel("residual stream depth")
    ax.set_title(f"Layer/depth sweep heatmap ({split})")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "layer_sweep_heatmap.png", "Depth-by-metric heatmap for probe and surface-control metrics.")
    add_manifest("layer_sweep_heatmap.png", "Where does the train-selected site sit relative to the full depth sweep?", heat_source, len(probe_rows), "probe and surface metrics", "surface/shuffled controls", "Depth evidence only supports the selected prompt-boundary readout.", "A bright heatmap cell is not a circuit.")

    # Probe by depth line plot, retained and improved.
    depth_source = write_figure_source(ctx, "tool_choice_probe_by_depth_source.csv", probe_rows, "Line-plot source for tool probe scores by depth and split.")
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for split_name in ("train", "eval"):
        rows = [r for r in probe_rows if r.get("split_group") == split_name]
        if not rows:
            continue
        ax.plot([int(r["depth"]) for r in rows], [as_float(r.get("tool_needed_auc"), 0.0) for r in rows], marker="o", label=f"{split_name} needed AUC")
        ax.plot([int(r["depth"]) for r in rows], [as_float(r.get("tool_selection_accuracy"), 0.0) for r in rows], marker="s", label=f"{split_name} tool acc")
    surface_rows = [r for r in probe_rows if r.get("split_group") == "eval"]
    if surface_rows:
        ax.plot([int(r["depth"]) for r in surface_rows], [as_float(r.get("surface_control_accuracy"), 0.0) for r in surface_rows], marker="^", label="eval surface acc")
    if not probe_rows:
        add_empty_message(ax, "No probe rows")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("residual stream depth")
    ax.set_ylabel("score")
    ax.set_title("Tool-choice probe by depth")
    ax.legend(fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "tool_choice_probe_by_depth.png", "Tool probe score by residual depth and split.")
    add_manifest("tool_choice_probe_by_depth.png", "Do probe scores survive eval rather than only train?", depth_source, len(probe_rows), "AUC/accuracy by depth", "surface and shuffled controls", "Decode claim only at the selected eval cell.", "Do not choose depth by the eval curve.")

    # ------------------------------------------------------------------
    # Confusion matrix.
    # ------------------------------------------------------------------
    confusion_rows_source: list[dict[str, Any]] = []
    labels = list(TOOLS)
    idx = {tool: i for i, tool in enumerate(labels)}
    split = "eval" if any(str(r.get("split")) == "eval" for r in task_rows) else "all"
    selected_tasks = [r for r in task_rows if split == "all" or str(r.get("split")) == split]
    matrix = np.zeros((len(labels), len(labels)))
    for row in selected_tasks:
        required = str(row.get("required_tool"))
        pred = str(row.get("tool_probe_prediction"))
        if required in idx and pred in idx:
            matrix[idx[required]][idx[pred]] += 1
    for required in labels:
        for pred in labels:
            confusion_rows_source.append({"split_group": split, "required_tool": required, "predicted_tool": pred, "count": int(matrix[idx[required], idx[pred]])})
    confusion_source = write_figure_source(ctx, "tool_selection_confusion_matrix_source.csv", confusion_rows_source, "Confusion-matrix source counts for selected-depth predictions.")
    fig, ax = plt.subplots(figsize=(7.4, 6.3))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("required")
    ax.set_title(f"Tool-selection confusion matrix ({split})")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j]:
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "tool_selection_confusion_matrix.png", "Required-vs-predicted tool confusion matrix.")
    add_manifest("tool_selection_confusion_matrix.png", "Which tool confusions shrink the which-tool claim?", confusion_source, len(confusion_rows_source), "count", "required-tool labels", "Positive language requires confusions to be bounded.", "No-tool false positives matter more than pretty diagonals.")

    # ------------------------------------------------------------------
    # Surface control ladder.
    # ------------------------------------------------------------------
    selected_depths = {int(r.get("selected_depth", 0)) for r in task_rows if str(r.get("selected_depth", "")).strip()}
    selected_depth = next(iter(selected_depths), 0)
    selected = selected_probe_row(probe_rows, selected_depth, "eval")
    surface_source_rows = [
        {"metric": "residual_probe_eval_accuracy", "value": safe_mean([1.0 if boolish(r.get("tool_probe_correct")) else 0.0 for r in task_rows if r.get("split") == "eval"], 0.0), "source": "task_manifest"},
        {"metric": "surface_heuristic_eval_accuracy", "value": safe_mean([1.0 if boolish(r.get("surface_cue_correct")) else 0.0 for r in task_rows if r.get("split") == "eval"], 0.0), "source": "task_manifest"},
        {"metric": "shuffled_label_control_accuracy", "value": as_float(selected.get("shuffled_label_control_accuracy"), 0.0) if selected else 0.0, "source": "tool_choice_probe_report"},
        {"metric": "no_tool_false_positive_rate", "value": as_float(selected.get("no_tool_false_positive_rate"), 0.0) if selected else 0.0, "source": "tool_choice_probe_report"},
    ]
    surface_source = write_figure_source(ctx, "surface_control_ladder_source.csv", surface_source_rows, "Source values for surface/shuffled/no-tool control ladder.")
    fig, ax = plt.subplots(figsize=(8.1, 4.8))
    ax.bar([str(r["metric"]).replace("_", "\n") for r in surface_source_rows], [as_float(r.get("value"), 0.0) for r in surface_source_rows])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate / accuracy")
    ax.set_title("Surface control ladder")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "surface_control_ladder.png", "Probe accuracy versus surface, shuffled, and no-tool controls.")
    add_manifest("surface_control_ladder.png", "Does residual decoding beat the boring lexical heuristic?", surface_source, len(surface_source_rows), "accuracy/rate", "surface and shuffled controls", "Supported only when residual probe beats surface on eval.", "If surface wins, the negative result is the lesson.")

    # ------------------------------------------------------------------
    # Trace trajectory and trace audit plots.
    # ------------------------------------------------------------------
    trajectory_source = write_figure_source(ctx, "trajectory_source.csv", transition_rows, "Tool-state transition rows used by the trajectory plot.")
    steps = sorted({int(r.get("step_index", 0)) for r in transition_rows})
    tools = list(TOOLS)
    step_matrix = np.zeros((len(tools), len(steps))) if steps else np.zeros((len(tools), 1))
    step_to_i = {step: i for i, step in enumerate(steps)}
    for row in transition_rows:
        tool = str(row.get("tool") or "none")
        if tool not in tools:
            tool = "none"
        step_matrix[tools.index(tool), step_to_i.get(int(row.get("step_index", 0)), 0)] += 1
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    if steps:
        im = ax.imshow(step_matrix, aspect="auto")
        ax.set_xticks(range(len(steps)), [str(s) for s in steps])
        ax.set_yticks(range(len(tools)), tools)
        ax.set_xlabel("trace step index")
        ax.set_ylabel("tool named in trace step")
        for i in range(step_matrix.shape[0]):
            for j in range(step_matrix.shape[1]):
                if step_matrix[i, j]:
                    ax.text(j, i, int(step_matrix[i, j]), ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
    else:
        add_empty_message(ax, "No transition rows")
    ax.set_title("Deterministic toy-tool trace trajectory")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "trajectory.png", "Trace trajectory from prompt receipt through local tool result.")
    add_manifest("trajectory.png", "What trace did the harness actually execute?", trajectory_source, len(transition_rows), "step/tool counts", "known deterministic simulator", "Trace audit only, not self-report.", "The model did not produce this trace.")

    mem_rows: list[dict[str, Any]] = []
    read_counts = defaultdict(int)
    for row in trace_rows:
        fam = str(row.get("family"))
        reads = safe_json_list(row.get("memory_reads_json"))
        read_counts[fam] += len(reads)
        mem_rows.append({"task_id": row.get("task_id"), "family": fam, "n_memory_reads": len(reads), "memory_reads_json": row.get("memory_reads_json")})
    mem_source = write_figure_source(ctx, "memory_read_trace_atlas_source.csv", mem_rows, "Memory-read counts derived from deterministic trace rows.")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    fams = sorted(read_counts)
    if fams:
        ax.bar(fams, [read_counts[f] for f in fams])
        ax.set_xticks(range(len(fams)), fams, rotation=35, ha="right")
    else:
        add_empty_message(ax, "No trace rows")
    ax.set_ylabel("known memory/tool reads")
    ax.set_title("Memory-read trace atlas")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "memory_read_trace_atlas.png", "Known memory-read trace counts by family.")
    add_manifest("memory_read_trace_atlas.png", "Which toy tools read synthetic memory objects?", mem_source, len(mem_rows), "memory read count", "closed local simulator", "Only a harness trace claim.", "Do not infer model memory reads.")

    reliance_rows: list[dict[str, Any]] = []
    reliance = defaultdict(list)
    for row in trace_rows:
        val = 1.0 if boolish(row.get("would_final_answer_change_if_tool_result_corrupted")) else 0.0
        reliance[str(row.get("family"))].append(val)
        reliance_rows.append({"task_id": row.get("task_id"), "family": row.get("family"), "would_change": val})
    reliance_source = write_figure_source(ctx, "tool_result_reliance_ladder_source.csv", reliance_rows, "Source rows for corrupted-result reliance by family.")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    fams = sorted(reliance)
    if fams:
        ax.bar(fams, [safe_mean(reliance[f], 0.0) for f in fams])
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(len(fams)), fams, rotation=35, ha="right")
    else:
        add_empty_message(ax, "No trace rows")
    ax.set_ylabel("fraction would change")
    ax.set_title("Tool-result reliance ladder")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "tool_result_reliance_ladder.png", "Toy final-answer reliance on tool result by family.")
    add_manifest("tool_result_reliance_ladder.png", "Would corrupting the toy tool result change the toy final answer?", reliance_source, len(reliance_rows), "fraction would change", "corrupted tool result", "Only a deterministic-trace reliance claim.", "Does not show the model verifies tool outputs.")

    self_report_source = write_figure_source(ctx, "tool_self_report_matrix_source.csv", report_rows, "Known-trace self-report review scaffold rows.")
    fig, ax = plt.subplots(figsize=(6.7, 4.8))
    match = sum(1 for r in report_rows if boolish(r.get("matches_known_trace")))
    review = sum(1 for r in report_rows if boolish(r.get("requires_human_review")))
    mat = np.array([[match, len(report_rows) - match], [review, len(report_rows) - review]])
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks([0, 1], ["yes", "no"])
    ax.set_yticks([0, 1], ["matches trace", "needs review"])
    ax.set_title("Tool self-report review matrix")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "tool_self_report_matrix.png", "Known-trace self-report review matrix.")
    add_manifest("tool_self_report_matrix.png", "Which rows require human review before self-report/source-attribution language?", self_report_source, len(report_rows), "review/match counts", "blank review columns", "Review scaffold only.", "No model self-report is generated by the default run.")

    # ------------------------------------------------------------------
    # Failure specimens / paired examples.
    # ------------------------------------------------------------------
    failure_source = write_figure_source(ctx, "paired_examples_source.csv", counterexamples, "Counterexample rows used for the paired_examples plot.")
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    failures = sorted(counterexamples, key=lambda r: as_float(r.get("severity"), 0.0), reverse=True)[:14]
    if failures:
        labels_y = [f"{r.get('kind')}\n{r.get('task_id')}" for r in failures]
        vals_y = [as_float(r.get("severity"), 0.0) for r in failures]
        ypos = np.arange(len(failures))
        ax.barh(ypos, vals_y)
        ax.set_yticks(ypos, labels_y, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("severity")
    else:
        add_empty_message(ax, "No automatic failure specimens crossed filters")
    ax.set_title("Failure specimens and counterexamples")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "paired_examples.png", "Counterexample specimen plot sorted by severity.")
    add_manifest("paired_examples.png", "Which concrete rows most weaken the favorite claim?", failure_source, len(counterexamples), "severity", "counterexample filters", "Counterexamples define claim boundaries.", "No automatic failures is not proof of no failures.")

    write_plot_manifest(ctx, manifest)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    tasks, data_info = load_tasks(ctx)
    manifest_path = ctx.path("diagnostics", "data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Lab 34 data manifest and toy-tool scope.")
    write_safety_status(ctx, data_info)

    tasks, letter_ids, token_rows = tokenization_gate(ctx, bundle, tasks)
    if not tasks:
        raise RuntimeError("Lab 34 selected zero tasks after tokenization.")

    bench.run_hook_parity_check(ctx, bundle, render_prompt(tasks[0]))
    first = bench.run_with_residual_cache(bundle, render_prompt(tasks[0]))
    bench.run_lens_self_check(ctx, bundle, first)
    bench.run_patch_noop_check(ctx, bundle, render_prompt(tasks[0]))

    depths = choose_depths(bundle, str(ctx.args.prompt_set))
    vectors, _prompt_audit_rows = capture_vectors(ctx, bundle, tasks, depths)
    models_by_depth = {depth: build_direction_model(ctx, bundle, tasks, vectors, depth) for depth in depths}
    probe_rows, selection_rows = build_probe_reports(tasks, vectors, models_by_depth)
    selected_depth = select_depth(selection_rows)
    selected_model = models_by_depth[selected_depth]
    for row in selection_rows:
        row["selected_depth"] = int(row["depth"]) == selected_depth
        row["selection_rule"] = "train decode gap over surface, then tool-needed AUC, claimable interior depth preferred"
    for row in probe_rows:
        row["selected_depth"] = int(row["depth"]) == selected_depth

    task_rows = task_manifest_rows(tasks, vectors, selected_model)
    surface_rows = surface_cue_audit_rows(tasks)
    confusion = confusion_rows(task_rows)
    trace_rows, transition_rows, self_report_rows, arg_rows = build_trace_tables(tasks)
    intervention_rows, intervention_summary = run_interventions(bundle, tasks, selected_model, letter_ids)
    counterexamples = build_counterexamples(task_rows, intervention_rows, trace_rows)
    evidence_rows, metrics = build_evidence_matrix(
        data_info,
        selected_depth,
        probe_rows,
        intervention_summary,
        trace_rows,
        self_report_rows,
        counterexamples,
    )
    metrics["directions"] = {"selected": selected_model.metadata, "depths_scanned": depths}
    metrics["self_check_status"] = write_self_check_status(ctx, token_rows, arg_rows, evidence_rows)
    run_config_snapshot = write_run_config_snapshot(ctx, data_info, selected_model, depths)
    warnings = write_warning_summary(ctx, data_info, task_rows, probe_rows, intervention_summary, trace_rows, evidence_rows, counterexamples)
    failure_specimens = write_failure_specimens(ctx, counterexamples, task_rows, intervention_rows, trace_rows)
    source_manifest = write_plot_source_tables_and_manifest(
        ctx,
        task_rows,
        probe_rows,
        selection_rows,
        surface_rows,
        confusion,
        intervention_rows,
        intervention_summary,
        trace_rows,
        transition_rows,
        self_report_rows,
        evidence_rows,
        counterexamples,
        arg_rows,
    )
    metrics["artifact_quality"] = {
        "run_config_snapshot_written": bool(run_config_snapshot),
        "warning_count": len(warnings),
        "failure_specimens": len(failure_specimens),
        "figure_source_tables": "written for every run, including --no-plots",
        "plot_manifest_entries": len(source_manifest),
    }

    write_tables(
        ctx,
        task_rows,
        probe_rows,
        selection_rows,
        surface_rows,
        confusion,
        intervention_rows,
        intervention_summary,
        trace_rows,
        transition_rows,
        self_report_rows,
        evidence_rows,
        counterexamples,
        arg_rows,
    )
    write_state(ctx, selected_model, {depth: model.metadata for depth, model in models_by_depth.items()})
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 34 metrics and verdicts.")
    write_method_card(ctx, data_info, selected_model, evidence_rows, metrics)
    write_operationalization_audit(ctx, evidence_rows, counterexamples)
    write_run_summary(ctx, data_info, metrics, evidence_rows)
    write_ledger_claims(ctx, evidence_rows)
    write_plots(ctx, task_rows, probe_rows, intervention_rows, intervention_summary, trace_rows, transition_rows, self_report_rows, evidence_rows, counterexamples)
    print(
        f"[lab34] selected depth {selected_depth}; wrote {len(task_rows)} task rows, "
        f"{len(probe_rows)} probe rows, {len(intervention_rows)} intervention rows, and {len(evidence_rows)} evidence rows"
    )
