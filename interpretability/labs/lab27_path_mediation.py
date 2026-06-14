"""Lab 27: Path-specific patching and causal mediation.

This lab compares ordinary residual-node patching with a residual
source-to-receiver mediation proxy. It is deliberately not full edge-level path
patching: the proxy asks whether a clean source patch changes a downstream
receiver state in a behaviorally useful way, and then makes the competing
node-only and wrong-route explanations visible in artifacts.
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
PATH_DEPTH_BUDGETS = {"small": (2, 3), "medium": (3, 4), "full": (4, 5)}
BASELINE_MARGIN = 0.20
MIN_DENOMINATOR = 0.40
MIN_MEDIATED_RECOVERY = 0.20
MIN_SPECIFICITY_GAP = 0.10
CONTROL_CLOSE_TOL = 0.05
MAX_COUNTEREXAMPLES = 32
REQUIRED_COLUMNS = {
    "item_id", "domain", "prompt", "clean_prompt", "corrupt_prompt", "target",
    "distractor", "positions_json", "candidate_nodes_json", "candidate_paths_json",
}


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
    raw_positions: dict[str, int] = dataclasses.field(default_factory=dict)
    token_offset: int = 0
    target_id: int = -1
    distractor_id: int = -1
    clean_ids: list[int] = dataclasses.field(default_factory=list)
    corrupt_ids: list[int] = dataclasses.field(default_factory=list)
    clean_diff: float = float("nan")
    corrupt_diff: float = float("nan")
    denominator: float = float("nan")
    baseline_pass: bool = False
    baseline_drop_reason: str = ""

    @property
    def source_pos(self) -> int:
        return int(self.positions["source_pos"])

    @property
    def receiver_pos(self) -> int:
        return int(self.positions.get("receiver_pos", self.positions["final_pos"]))

    @property
    def final_pos(self) -> int:
        return int(self.positions["final_pos"])

    @property
    def wrong_pos(self) -> int:
        return int(self.positions.get("wrong_pos", 0))

    @property
    def path_id(self) -> str:
        if self.candidate_paths and isinstance(self.candidate_paths[0], Mapping):
            return str(self.candidate_paths[0].get("path_id", "source_to_receiver"))
        return "source_to_receiver"


@dataclasses.dataclass(frozen=True)
class PatchCapture:
    input_ids: list[int]
    streams: Any
    final_logits_last: Any


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


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def rounded(value: Any, digits: int = 4) -> Any:
    val = fnum(value)
    return round(val, digits) if math.isfinite(val) else ""


def safe_mean(values: Sequence[Any], default: float = float("nan")) -> float:
    vals = [fnum(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(statistics.fmean(vals)) if vals else default


def safe_max(values: Sequence[Any], default: float = float("nan")) -> float:
    vals = [fnum(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return max(vals) if vals else default


def safe_corr(xs: Sequence[Any], ys: Sequence[Any]) -> float:
    pairs = [(fnum(x), fnum(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    xb = statistics.fmean(x for x, _ in pairs)
    yb = statistics.fmean(y for _, y in pairs)
    num = sum((x - xb) * (y - yb) for x, y in pairs)
    denx = math.sqrt(sum((x - xb) ** 2 for x, _ in pairs))
    deny = math.sqrt(sum((y - yb) ** 2 for _, y in pairs))
    return num / (denx * deny) if denx > 1e-12 and deny > 1e-12 else float("nan")


def logit_diff(logits: Any, task: PathTask) -> float:
    return float(logits[task.target_id] - logits[task.distractor_id])


def recovery(patched_diff: float, task: PathTask) -> float:
    if abs(task.denominator) < 1e-9:
        return float("nan")
    return (float(patched_diff) - task.corrupt_diff) / task.denominator


def visible_token_list(tokenizer: Any, ids: Sequence[int]) -> str:
    return " | ".join(f"{i}:{bench.visible_token(tokenizer.decode([int(t)]))}" for i, t in enumerate(ids))


def parse_json_cell(row: Mapping[str, str], column: str) -> Any:
    item_id = row.get("item_id", "<unknown>")
    try:
        return json.loads(row[column])
    except Exception as exc:
        raise ValueError(f"{item_id}: invalid {column}: {exc}") from exc


# ---------------------------------------------------------------------------
# Data and tokenization gates
# ---------------------------------------------------------------------------


def data_path(args: Any) -> pathlib.Path:
    prompt_set = str(getattr(args, "prompt_set", "") or "")
    candidate = pathlib.Path(prompt_set)
    if prompt_set not in PROMPT_SET_CAPS and candidate.suffix.lower() in {".csv", ".tsv"}:
        return candidate if candidate.is_absolute() else (bench.COURSE_ROOT / candidate).resolve()
    return bench.COURSE_ROOT / "data" / DATA_FILE


def balanced_cap(tasks: Sequence[PathTask], cap: int) -> list[PathTask]:
    if cap <= 0 or len(tasks) <= cap:
        return list(tasks)
    by_domain: dict[str, list[PathTask]] = defaultdict(list)
    for task in tasks:
        by_domain[task.domain].append(task)
    out: list[PathTask] = []
    cursor = 0
    domains = sorted(by_domain)
    while len(out) < cap:
        made_progress = False
        for domain in domains:
            if cursor < len(by_domain[domain]):
                out.append(by_domain[domain][cursor])
                made_progress = True
                if len(out) >= cap:
                    break
        if not made_progress:
            break
        cursor += 1
    return out


def manifest_expected_hash(path: pathlib.Path) -> tuple[str | None, str]:
    manifest_path = path.parent / "MANIFEST.json"
    if not manifest_path.exists():
        return None, "data/MANIFEST.json not found"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"data/MANIFEST.json unreadable: {exc}"
    candidates: list[Any] = []
    if isinstance(manifest, dict):
        candidates.extend([
            manifest.get(path.name),
            manifest.get(str(path)),
            manifest.get("files", {}).get(path.name) if isinstance(manifest.get("files"), dict) else None,
        ])
    for entry in candidates:
        if isinstance(entry, str):
            return entry, "found string entry"
        if isinstance(entry, dict):
            for key in ("sha256", "hash", "sha256_hex"):
                val = entry.get(key)
                if isinstance(val, str):
                    return val, f"found {key} entry"
    return None, f"no usable sha256 entry for {path.name}"


def parse_task(row: Mapping[str, str]) -> PathTask:
    positions = parse_json_cell(row, "positions_json")
    nodes = parse_json_cell(row, "candidate_nodes_json")
    paths = parse_json_cell(row, "candidate_paths_json")
    if not isinstance(positions, dict):
        raise ValueError(f"{row['item_id']}: positions_json must be an object")
    if not isinstance(nodes, dict):
        raise ValueError(f"{row['item_id']}: candidate_nodes_json must be an object")
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"{row['item_id']}: candidate_paths_json must be a non-empty list")
    pos = {str(k): int(v) for k, v in positions.items()}
    if "source_pos" not in pos or "final_pos" not in pos:
        raise ValueError(f"{row['item_id']}: positions_json needs source_pos and final_pos")
    pos.setdefault("receiver_pos", pos["final_pos"])
    pos.setdefault("wrong_pos", 0 if pos["source_pos"] != 0 else 1)
    return PathTask(
        item_id=row["item_id"].strip(),
        domain=row["domain"].strip(),
        prompt=row.get("prompt", row["clean_prompt"]),
        clean_prompt=row["clean_prompt"],
        corrupt_prompt=row["corrupt_prompt"],
        target=row["target"],
        distractor=row["distractor"],
        positions=dict(pos),
        raw_positions=dict(pos),
        candidate_nodes=dict(nodes),
        candidate_paths=list(paths),
    )


def load_tasks(ctx: bench.RunContext) -> tuple[list[PathTask], dict[str, Any]]:
    path = data_path(ctx.args)
    if not path.exists():
        raise FileNotFoundError(f"Lab 27 data file not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} contains no rows")
    missing = sorted(REQUIRED_COLUMNS - set(rows[0]))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    all_tasks = [parse_task(row) for row in rows]
    selected = balanced_cap(all_tasks, PROMPT_SET_CAPS.get(str(ctx.args.prompt_set), 0))
    if int(ctx.args.max_examples or 0) > 0:
        selected = balanced_cap(selected, int(ctx.args.max_examples))
    actual_sha = file_sha256(path)
    expected_sha, manifest_note = manifest_expected_hash(path)
    info = {
        "data_file": DATA_FILE,
        "data_path": str(path),
        "data_sha256": actual_sha,
        "manifest_expected_sha256": expected_sha,
        "manifest_note": manifest_note,
        "manifest_ok": (actual_sha == expected_sha) if expected_sha else None,
        "n_rows_file": len(all_tasks),
        "n_rows_selected": len(selected),
        "domains_selected": {d: sum(1 for t in selected if t.domain == d) for d in sorted({t.domain for t in selected})},
        "prompt_set": ctx.args.prompt_set,
        "max_examples": ctx.args.max_examples,
        "science_ready": True,
        "fallback_data": False,
    }
    return selected, info


def find_subsequence_offset(raw: Sequence[int], runtime: Sequence[int]) -> int | None:
    raw_ids, runtime_ids = list(raw), list(runtime)
    for start in range(len(runtime_ids) - len(raw_ids) + 1):
        if runtime_ids[start:start + len(raw_ids)] == raw_ids:
            return start
    return None


def tokenization_gate(ctx: bench.RunContext, bundle: bench.ModelBundle, tasks: list[PathTask]) -> tuple[list[PathTask], list[dict[str, Any]]]:
    tok = bundle.tokenizer
    kept: list[PathTask] = []
    rows: list[dict[str, Any]] = []
    for task in tasks:
        problems: list[str] = []
        raw_clean = tok.encode(task.clean_prompt, add_special_tokens=False)
        raw_corrupt = tok.encode(task.corrupt_prompt, add_special_tokens=False)
        runtime_clean = tok(task.clean_prompt, add_special_tokens=True)["input_ids"]
        runtime_corrupt = tok(task.corrupt_prompt, add_special_tokens=True)["input_ids"]
        target_ids = tok.encode(task.target, add_special_tokens=False)
        distractor_ids = tok.encode(task.distractor, add_special_tokens=False)
        if len(raw_clean) != len(raw_corrupt):
            problems.append("raw_clean_corrupt_length_mismatch")
        if len(runtime_clean) != len(runtime_corrupt):
            problems.append("runtime_clean_corrupt_length_mismatch")
        if len(target_ids) != 1:
            problems.append(f"target_tokens={len(target_ids)}")
        if len(distractor_ids) != 1:
            problems.append(f"distractor_tokens={len(distractor_ids)}")
        if len(target_ids) == 1 and len(distractor_ids) == 1 and target_ids[0] == distractor_ids[0]:
            problems.append("target_equals_distractor_token")
        clean_offset = find_subsequence_offset(raw_clean, runtime_clean)
        corrupt_offset = find_subsequence_offset(raw_corrupt, runtime_corrupt)
        if clean_offset is None or corrupt_offset is None:
            problems.append("raw_ids_not_subsequence_of_runtime_ids")
            offset = 0
        elif clean_offset != corrupt_offset:
            problems.append(f"clean_corrupt_token_offset_mismatch:{clean_offset}!={corrupt_offset}")
            offset = clean_offset
        else:
            offset = clean_offset
        adjusted = {k: int(v) + int(offset) for k, v in task.raw_positions.items()}
        adjusted.setdefault("receiver_pos", adjusted["final_pos"])
        adjusted.setdefault("wrong_pos", 0 if adjusted["source_pos"] != 0 else 1)
        for name, pos in adjusted.items():
            if not 0 <= int(pos) < len(runtime_clean):
                problems.append(f"{name}_out_of_range:{pos}")
        if adjusted.get("source_pos") == adjusted.get("wrong_pos"):
            problems.append("wrong_pos_equals_source_pos")
        if adjusted.get("receiver_pos") == adjusted.get("wrong_pos"):
            problems.append("wrong_pos_equals_receiver_pos")
        diff_positions = [i for i, (a, b) in enumerate(zip(runtime_clean, runtime_corrupt)) if a != b]
        if adjusted.get("source_pos") not in diff_positions:
            problems.append("source_pos_not_a_clean_corrupt_difference")
        for i, spec in enumerate(task.candidate_paths):
            if isinstance(spec, Mapping):
                for key in ("source_position", "receiver_position"):
                    val = str(spec.get(key, ""))
                    if val and val not in adjusted:
                        problems.append(f"candidate_paths[{i}].{key}={val!r}_not_in_positions")
            else:
                problems.append(f"candidate_paths[{i}]_not_object")
        task.positions = adjusted
        task.token_offset = int(offset)
        task.clean_ids = [int(x) for x in runtime_clean]
        task.corrupt_ids = [int(x) for x in runtime_corrupt]
        if not problems:
            task.target_id = int(target_ids[0])
            task.distractor_id = int(distractor_ids[0])
            kept.append(task)
        rows.append({
            "item_id": task.item_id,
            "domain": task.domain,
            "kept": not problems,
            "problems": ";".join(problems),
            "token_offset_applied": int(offset),
            "raw_n_tokens_clean": len(raw_clean),
            "runtime_n_tokens_clean": len(runtime_clean),
            "runtime_n_tokens_corrupt": len(runtime_corrupt),
            "target": bench.visible_token(task.target),
            "target_token_count": len(target_ids),
            "distractor": bench.visible_token(task.distractor),
            "distractor_token_count": len(distractor_ids),
            "source_pos": adjusted.get("source_pos", ""),
            "receiver_pos": adjusted.get("receiver_pos", ""),
            "final_pos": adjusted.get("final_pos", ""),
            "wrong_pos": adjusted.get("wrong_pos", ""),
            "diff_positions": " ".join(str(i) for i in diff_positions),
            "clean_tokens": visible_token_list(tok, runtime_clean),
            "corrupt_tokens": visible_token_list(tok, runtime_corrupt),
        })
    path = ctx.path("diagnostics", "tokenization_gate.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic", "Runtime token alignment and position audit for Lab 27 tasks.")
    if not kept:
        raise RuntimeError("Lab 27 tokenization gate dropped every task.")
    return kept, rows


# ---------------------------------------------------------------------------
# Residual patching machinery
# ---------------------------------------------------------------------------


def module_for_depth(bundle: bench.ModelBundle, depth: int) -> Any:
    if not 0 <= int(depth) <= bundle.anatomy.n_layers:
        raise ValueError(f"depth must be in [0, {bundle.anatomy.n_layers}], got {depth}")
    return bundle.final_norm if int(depth) == bundle.anatomy.n_layers else bundle.blocks[int(depth)]


def patch_hooks(bundle: bench.ModelBundle, patches: Sequence[tuple[int, int, Any]]) -> list[tuple[Any, Any]]:
    grouped: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for depth, pos, vec in patches:
        grouped[int(depth)].append((int(pos), vec))
    hooks: list[tuple[Any, Any]] = []
    for depth, entries in sorted(grouped.items()):
        module = module_for_depth(bundle, depth)

        def make_hook(layer_entries: list[tuple[int, Any]], layer_depth: int):
            def hook(mod: Any, hook_args: tuple) -> Any:
                del mod
                hidden = hook_args[0].clone()
                seq_len = hidden.shape[1]
                for pos, vec in layer_entries:
                    if not -seq_len <= pos < seq_len:
                        raise ValueError(f"position {pos} out of range at depth {layer_depth}")
                    hidden[0, pos] = vec.to(hidden.device, hidden.dtype)
                return (hidden,) + tuple(hook_args[1:])
            return hook

        hooks.append((module, make_hook(list(entries), depth)))
    return hooks


def run_multi_site_patch(bundle: bench.ModelBundle, prompt: str, patches: Sequence[tuple[int, int, Any]]) -> Any:
    return bench._forward_logits(bundle, prompt, patch_hooks(bundle, patches))


def run_with_patch_capture(bundle: bench.ModelBundle, prompt: str, patches: Sequence[tuple[int, int, Any]]) -> PatchCapture:
    import torch

    encoded = bundle.tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(bundle.input_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.input_device)
    final_capture: dict[str, Any] = {}

    def final_pre_hook(module: Any, hook_args: tuple) -> None:
        del module
        final_capture["final_prenorm"] = bench.tensor_cpu_float(hook_args[0])

    handles = []
    try:
        for module, hook in patch_hooks(bundle, patches):
            handles.append(module.register_forward_pre_hook(hook))
        handles.append(bundle.final_norm.register_forward_pre_hook(final_pre_hook))
        with torch.no_grad():
            out = bundle.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if "final_prenorm" not in final_capture:
        raise RuntimeError("final-norm pre-hook did not fire during patched capture")
    n_layers = bundle.anatomy.n_layers
    if len(out.hidden_states) != n_layers + 1:
        raise RuntimeError(f"expected {n_layers + 1} hidden states, got {len(out.hidden_states)}")
    streams = torch.stack([bench.tensor_cpu_float(h[0]) for h in out.hidden_states[:-1]] + [final_capture["final_prenorm"][0]])
    return PatchCapture(input_ids=input_ids[0].detach().cpu().tolist(), streams=streams, final_logits_last=bench.tensor_cpu_float(out.logits[0, -1]))


# ---------------------------------------------------------------------------
# Measurement pipeline
# ---------------------------------------------------------------------------


def depth_grid(n_layers: int, prompt_set: str) -> list[int]:
    if prompt_set == "full":
        return list(range(n_layers + 1))
    if prompt_set == "medium":
        anchors = {0, n_layers, max(1, n_layers // 4), max(1, n_layers // 2), max(1, 3 * n_layers // 4)}
        step = max(1, n_layers // 8)
        anchors.update(range(1, n_layers, step))
        return sorted(d for d in anchors if 0 <= d <= n_layers)
    return sorted({0, max(1, n_layers // 4), max(1, n_layers // 2), max(1, 3 * n_layers // 4), n_layers})


def baseline_pass(task: PathTask) -> tuple[bool, str]:
    reasons: list[str] = []
    if not math.isfinite(task.clean_diff) or not math.isfinite(task.corrupt_diff):
        reasons.append("nonfinite_baseline")
    if task.clean_diff <= BASELINE_MARGIN:
        reasons.append(f"clean_diff<={BASELINE_MARGIN}")
    if task.corrupt_diff >= -BASELINE_MARGIN:
        reasons.append(f"corrupt_diff>=-{BASELINE_MARGIN}")
    if task.denominator <= MIN_DENOMINATOR:
        reasons.append(f"denominator<={MIN_DENOMINATOR}")
    return not reasons, ";".join(reasons)


def cache_baselines(ctx: bench.RunContext, bundle: bench.ModelBundle, tasks: list[PathTask]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    clean_caps: dict[str, Any] = {}
    corrupt_caps: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for i, task in enumerate(tasks):
        clean = bench.run_with_residual_cache(bundle, task.clean_prompt)
        corrupt = bench.run_with_residual_cache(bundle, task.corrupt_prompt)
        if clean.input_ids != task.clean_ids or corrupt.input_ids != task.corrupt_ids:
            raise RuntimeError(f"{task.item_id}: capture tokenization differs from tokenization gate")
        clean_caps[task.item_id] = clean
        corrupt_caps[task.item_id] = corrupt
        task.clean_diff = logit_diff(clean.final_logits_last, task)
        task.corrupt_diff = logit_diff(corrupt.final_logits_last, task)
        task.denominator = task.clean_diff - task.corrupt_diff
        task.baseline_pass, task.baseline_drop_reason = baseline_pass(task)
        rows.append({
            "item_id": task.item_id,
            "domain": task.domain,
            "path_id": task.path_id,
            "clean_diff": rounded(task.clean_diff),
            "corrupt_diff": rounded(task.corrupt_diff),
            "denominator": rounded(task.denominator),
            "baseline_pass": task.baseline_pass,
            "drop_reason": task.baseline_drop_reason,
            "clean_top_token": bench.visible_token(bundle.tokenizer.decode([int(clean.final_logits_last.argmax())])),
            "corrupt_top_token": bench.visible_token(bundle.tokenizer.decode([int(corrupt.final_logits_last.argmax())])),
            "target": bench.visible_token(task.target),
            "distractor": bench.visible_token(task.distractor),
            "source_pos": task.source_pos,
            "receiver_pos": task.receiver_pos,
            "final_pos": task.final_pos,
            "wrong_pos": task.wrong_pos,
        })
        if (i + 1) % max(1, len(tasks) // 4) == 0:
            print(f"[lab27] cached baselines {i + 1}/{len(tasks)}")
    path = ctx.path("tables", "baseline_behavior.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Clean/corrupt baseline margins and behavior-gate status.")
    return clean_caps, corrupt_caps, rows


def run_node_screen(ctx: bench.RunContext, bundle: bench.ModelBundle, tasks: list[PathTask], clean_caps: Mapping[str, Any], depths: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eligible = [t for t in tasks if t.baseline_pass]
    total = max(1, len(eligible) * len(depths) * 3)
    done = 0
    for task in eligible:
        clean = clean_caps[task.item_id]
        for depth in depths:
            cells = [
                ("source", task.source_pos, clean.streams[int(depth), task.source_pos]),
                ("receiver", task.receiver_pos, clean.streams[int(depth), task.receiver_pos]),
                ("wrong_position", task.wrong_pos, clean.streams[int(depth), task.wrong_pos]),
            ]
            for node, pos, vec in cells:
                logits = bench.run_with_residual_patch(bundle, task.corrupt_prompt, int(depth), int(pos), vec)
                patched = logit_diff(logits, task)
                rec = recovery(patched, task)
                rows.append({
                    "item_id": task.item_id,
                    "domain": task.domain,
                    "path_id": task.path_id,
                    "node": node,
                    "depth": int(depth),
                    "position": int(pos),
                    "patched_diff": rounded(patched),
                    "recovery": rounded(rec),
                    "clean_diff": rounded(task.clean_diff),
                    "corrupt_diff": rounded(task.corrupt_diff),
                    "baseline_pass": task.baseline_pass,
                })
                done += 1
                if done % max(1, total // 5) == 0:
                    print(f"[lab27] node screen {done}/{total}")
    if eligible:
        print(f"[lab27] node screen {done}/{total}")
    return rows


def recovery_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int], float]:
    return {(str(r["item_id"]), str(r["node"]), int(r["depth"])): fnum(r.get("recovery")) for r in rows}


def select_path_depths(task: PathTask, node_rows: Sequence[Mapping[str, Any]], depths: Sequence[int], n_layers: int, prompt_set: str) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    n_source, n_receiver = PATH_DEPTH_BUDGETS.get(prompt_set, PATH_DEPTH_BUDGETS["small"])
    task_rows = [r for r in node_rows if r.get("item_id") == task.item_id]

    def ranked(node: str) -> list[int]:
        vals: list[tuple[int, float]] = []
        for r in task_rows:
            if r.get("node") != node:
                continue
            depth = int(r["depth"])
            if node == "source" and depth >= n_layers:
                continue
            if node == "receiver" and depth <= 0:
                continue
            rec = fnum(r.get("recovery"))
            if math.isfinite(rec):
                vals.append((depth, rec))
        return [d for d, _ in sorted(vals, key=lambda x: (x[1], -abs(x[0] - n_layers / 2)), reverse=True)]

    source_depths = ranked("source")[:n_source]
    receiver_depths = ranked("receiver")[:n_receiver]
    for d in (0, max(1, n_layers // 3), max(1, 2 * n_layers // 3)):
        if len(source_depths) < n_source and d in depths and d < n_layers and d not in source_depths:
            source_depths.append(d)
    for d in (max(1, n_layers // 3), max(1, 2 * n_layers // 3), n_layers):
        if len(receiver_depths) < n_receiver and d in depths and d > 0 and d not in receiver_depths:
            receiver_depths.append(d)
    lookup = recovery_lookup(task_rows)
    rows = []
    for rank, d in enumerate(sorted(set(source_depths)), start=1):
        rows.append({"item_id": task.item_id, "domain": task.domain, "node": "source", "selected_depth": d, "rank": rank, "node_recovery": rounded(lookup.get((task.item_id, "source", d))), "reason": "top_node_screen_or_anchor"})
    for rank, d in enumerate(sorted(set(receiver_depths)), start=1):
        rows.append({"item_id": task.item_id, "domain": task.domain, "node": "receiver", "selected_depth": d, "rank": rank, "node_recovery": rounded(lookup.get((task.item_id, "receiver", d))), "reason": "top_node_screen_or_anchor"})
    return sorted(set(source_depths)), sorted(set(receiver_depths)), rows


def random_depth(depths: Sequence[int], task: PathTask, key: str, *, less_than: int) -> int:
    candidates = [int(d) for d in depths if int(d) < less_than]
    if not candidates:
        candidates = [int(d) for d in depths]
    return candidates[stable_int(f"{task.item_id}|{key}") % len(candidates)]


def run_path_interventions(ctx: bench.RunContext, bundle: bench.ModelBundle, tasks: list[PathTask], clean_caps: Mapping[str, Any], node_rows: Sequence[Mapping[str, Any]], depths: Sequence[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    node_rec = recovery_lookup(node_rows)
    path_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    accounting_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    plans: list[tuple[PathTask, list[int], list[int]]] = []
    for task in [t for t in tasks if t.baseline_pass]:
        srcs, recvs, rows = select_path_depths(task, node_rows, depths, bundle.anatomy.n_layers, str(ctx.args.prompt_set))
        selection_rows.extend(rows)
        plans.append((task, srcs, recvs))
    total = max(1, sum(1 for _t, srcs, recvs in plans for s in srcs for r in recvs if s < r))
    done = 0
    for task, source_depths, receiver_depths in plans:
        clean = clean_caps[task.item_id]
        source_caps: dict[int, PatchCapture] = {
            s: run_with_patch_capture(bundle, task.corrupt_prompt, [(s, task.source_pos, clean.streams[s, task.source_pos])])
            for s in source_depths
        }
        random_caps: dict[tuple[int, int], PatchCapture] = {}
        for src_depth in source_depths:
            if source_caps[src_depth].input_ids != task.corrupt_ids:
                raise RuntimeError(f"{task.item_id}: patched capture tokenization drifted")
            for recv_depth in receiver_depths:
                if src_depth >= recv_depth:
                    continue
                clean_source = clean.streams[src_depth, task.source_pos]
                clean_receiver = clean.streams[recv_depth, task.receiver_pos]
                mediated_receiver = source_caps[src_depth].streams[recv_depth, task.receiver_pos]
                mediated_wrong = source_caps[src_depth].streams[recv_depth, task.wrong_pos]
                source_node = node_rec.get((task.item_id, "source", src_depth), float("nan"))
                receiver_node = node_rec.get((task.item_id, "receiver", recv_depth), float("nan"))
                wrong_node = node_rec.get((task.item_id, "wrong_position", recv_depth), float("nan"))

                mediated_logits = bench.run_with_residual_patch(bundle, task.corrupt_prompt, recv_depth, task.receiver_pos, mediated_receiver)
                mediated_diff = logit_diff(mediated_logits, task)
                mediated = recovery(mediated_diff, task)
                joint_logits = run_multi_site_patch(bundle, task.corrupt_prompt, [(src_depth, task.source_pos, clean_source), (recv_depth, task.receiver_pos, clean_receiver)])
                joint_diff = logit_diff(joint_logits, task)
                joint = recovery(joint_diff, task)
                wrong_logits = bench.run_with_residual_patch(bundle, task.corrupt_prompt, recv_depth, task.wrong_pos, mediated_wrong)
                wrong_control = recovery(logit_diff(wrong_logits, task), task)
                rand_src_depth = random_depth(depths, task, f"random_source|{src_depth}|{recv_depth}", less_than=recv_depth)
                key = (rand_src_depth, recv_depth)
                if key not in random_caps:
                    random_caps[key] = run_with_patch_capture(bundle, task.corrupt_prompt, [(rand_src_depth, task.wrong_pos, clean.streams[rand_src_depth, task.wrong_pos])])
                random_vec = random_caps[key].streams[recv_depth, task.receiver_pos]
                random_control = recovery(logit_diff(bench.run_with_residual_patch(bundle, task.corrupt_prompt, recv_depth, task.receiver_pos, random_vec), task), task)
                reverse_logits = run_multi_site_patch(bundle, task.corrupt_prompt, [(src_depth, task.receiver_pos, clean.streams[src_depth, task.receiver_pos]), (recv_depth, task.source_pos, clean.streams[recv_depth, task.source_pos])])
                reverse_control = recovery(logit_diff(reverse_logits, task), task)
                controls = {
                    "wrong_receiver_from_source_patch": wrong_control,
                    "random_source_to_receiver": random_control,
                    "reverse_site_two_site": reverse_control,
                }
                control_floor = safe_max(list(controls.values()))
                specificity_gap = mediated - control_floor if math.isfinite(control_floor) else float("nan")
                best_node = safe_max([source_node, receiver_node])
                joint_increment = joint - best_node if math.isfinite(best_node) else float("nan")
                interaction = joint - source_node - receiver_node
                common = {
                    "item_id": task.item_id,
                    "domain": task.domain,
                    "path_id": task.path_id,
                    "source_depth": src_depth,
                    "receiver_depth": recv_depth,
                    "source_position": task.source_pos,
                    "receiver_position": task.receiver_pos,
                    "wrong_position": task.wrong_pos,
                    "clean_diff": rounded(task.clean_diff),
                    "corrupt_diff": rounded(task.corrupt_diff),
                    "denominator": rounded(task.denominator),
                    "baseline_pass": task.baseline_pass,
                }
                path_rows.append({**common, "source_node_recovery": rounded(source_node), "receiver_node_recovery": rounded(receiver_node), "wrong_position_node_recovery": rounded(wrong_node), "mediated_path_recovery": rounded(mediated), "mediated_patched_diff": rounded(mediated_diff), "joint_clean_two_site_recovery": rounded(joint), "joint_patched_diff": rounded(joint_diff), "control_floor": rounded(control_floor), "specificity_gap": rounded(specificity_gap), "joint_increment_over_best_node": rounded(joint_increment), "interaction_residual": rounded(interaction), "mediated_over_best_node": rounded(mediated - best_node if math.isfinite(best_node) else float("nan")), "temporal_order_valid": True})
                for control, rec in controls.items():
                    control_rows.append({**common, "control": control, "control_recovery": rounded(rec), "path_control_gap": rounded(mediated - rec), "joint_control_gap": rounded(joint - rec), "control_matches_path": bool(math.isfinite(rec) and rec >= mediated - CONTROL_CLOSE_TOL)})
                accounting_rows.append({**common, "source_effect": rounded(source_node), "receiver_effect": rounded(receiver_node), "mediated_path_effect": rounded(mediated), "joint_clean_two_site_effect": rounded(joint), "control_floor": rounded(control_floor), "specificity_gap": rounded(specificity_gap), "joint_increment_over_best_node": rounded(joint_increment), "interaction_residual": rounded(interaction), "mediated_share_of_receiver_node": rounded(mediated / receiver_node if abs(receiver_node) > 1e-9 else float("nan")), "mediated_share_of_joint": rounded(mediated / joint if abs(joint) > 1e-9 else float("nan")), "node_dominance_gap": rounded(best_node - mediated if math.isfinite(best_node) else float("nan"))})
                done += 1
                if done % max(1, total // 5) == 0:
                    print(f"[lab27] path cells {done}/{total}")
    print(f"[lab27] path cells {done}/{total}")
    return path_rows, control_rows, accounting_rows, selection_rows


# ---------------------------------------------------------------------------
# Summaries and artifact writing
# ---------------------------------------------------------------------------


def classify_row(row: Mapping[str, Any]) -> tuple[str, str]:
    med = fnum(row.get("mediated_path_recovery"))
    gap = fnum(row.get("specificity_gap"))
    ctrl = fnum(row.get("control_floor"))
    src = fnum(row.get("source_node_recovery"))
    recv = fnum(row.get("receiver_node_recovery"))
    best_node = safe_max([src, recv])
    joint_inc = fnum(row.get("joint_increment_over_best_node"))
    if math.isfinite(ctrl) and ctrl >= med - CONTROL_CLOSE_TOL:
        return "failed_controls", "a control matched the mediated receiver proxy"
    if med >= MIN_MEDIATED_RECOVERY and gap >= MIN_SPECIFICITY_GAP:
        if math.isfinite(joint_inc) and joint_inc > MIN_SPECIFICITY_GAP:
            return "path_proxy_plus_joint_interaction", "mediated proxy and joint increment both cleared gates"
        return "path_proxy_supported", "mediated proxy beat wrong-site, random-source, and reverse-site controls"
    if math.isfinite(best_node) and best_node >= MIN_MEDIATED_RECOVERY and med < best_node:
        return "node_effect_only", "ordinary node patching is stronger than the path proxy"
    return "needs_refinement_or_negative", "path proxy did not clear recovery and specificity gates"


def summarize(tasks: Sequence[PathTask], node_rows: Sequence[Mapping[str, Any]], path_rows: Sequence[Mapping[str, Any]], control_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    counterexamples: list[dict[str, Any]] = []
    for domain in sorted({t.domain for t in tasks}):
        domain_tasks = [t for t in tasks if t.domain == domain]
        eligible = [t for t in domain_tasks if t.baseline_pass]
        rows = [r for r in path_rows if r.get("domain") == domain]
        nodes = [r for r in node_rows if r.get("domain") == domain]
        if not rows:
            posture = "behavior_gate_failed" if not eligible else "no_temporal_path_cells"
            evidence.append({"domain": domain, "n_tasks": len(domain_tasks), "n_behavior_pass": len(eligible), "n_path_cells": 0, "best_item": "", "path_id": "", "source_depth": "", "receiver_depth": "", "mean_source_node_recovery": rounded(safe_mean([r.get("recovery") for r in nodes if r.get("node") == "source"])), "mean_receiver_node_recovery": rounded(safe_mean([r.get("recovery") for r in nodes if r.get("node") == "receiver"])), "mean_mediated_path_recovery": "", "mean_specificity_gap": "", "best_source_node_recovery": "", "best_receiver_node_recovery": "", "best_mediated_path_recovery": "", "best_joint_clean_two_site_recovery": "", "best_joint_increment_over_best_node": "", "control_floor": "", "specificity_gap": "", "control_match_rate": "", "claim_posture": posture, "smallest_supported_claim": "No path claim. Inspect baseline_behavior.csv or depth_selection.csv.", "primary_caveat": posture})
            for task in domain_tasks:
                if not task.baseline_pass:
                    counterexamples.append({"domain": domain, "item_id": task.item_id, "kind": "behavior_gate_failed", "path_id": task.path_id, "source_depth": "", "receiver_depth": "", "path_recovery": "", "control_recovery": "", "control": "", "lesson": f"Clean/corrupt baseline did not create a meaningful denominator: {task.baseline_drop_reason}."})
            continue

        def score(row: Mapping[str, Any]) -> float:
            return fnum(row.get("specificity_gap"), -999.0) + 0.5 * fnum(row.get("mediated_path_recovery"), -999.0) + 0.25 * max(0.0, fnum(row.get("joint_increment_over_best_node"), 0.0))

        best = max(rows, key=score)
        posture, caveat = classify_row(best)
        same_controls = [c for c in control_rows if c.get("domain") == domain and c.get("item_id") == best.get("item_id") and int(c.get("source_depth")) == int(best.get("source_depth")) and int(c.get("receiver_depth")) == int(best.get("receiver_depth"))]
        close_controls = [c for c in same_controls if c.get("control_matches_path")]
        source_vals = [r.get("recovery") for r in nodes if r.get("node") == "source"]
        receiver_vals = [r.get("recovery") for r in nodes if r.get("node") == "receiver"]
        control_match_rate = safe_mean([1.0 if c.get("control_matches_path") else 0.0 for c in control_rows if c.get("domain") == domain])
        supported = posture in {"path_proxy_supported", "path_proxy_plus_joint_interaction"}
        evidence.append({"domain": domain, "n_tasks": len(domain_tasks), "n_behavior_pass": len(eligible), "n_path_cells": len(rows), "best_item": best.get("item_id", ""), "path_id": best.get("path_id", ""), "source_depth": best.get("source_depth", ""), "receiver_depth": best.get("receiver_depth", ""), "mean_source_node_recovery": rounded(safe_mean(source_vals)), "mean_receiver_node_recovery": rounded(safe_mean(receiver_vals)), "mean_mediated_path_recovery": rounded(safe_mean([r.get("mediated_path_recovery") for r in rows])), "mean_specificity_gap": rounded(safe_mean([r.get("specificity_gap") for r in rows])), "mean_joint_increment_over_best_node": rounded(safe_mean([r.get("joint_increment_over_best_node") for r in rows])), "best_source_node_recovery": best.get("source_node_recovery", ""), "best_receiver_node_recovery": best.get("receiver_node_recovery", ""), "best_mediated_path_recovery": best.get("mediated_path_recovery", ""), "best_joint_clean_two_site_recovery": best.get("joint_clean_two_site_recovery", ""), "best_joint_increment_over_best_node": best.get("joint_increment_over_best_node", ""), "control_floor": best.get("control_floor", ""), "specificity_gap": best.get("specificity_gap", ""), "control_match_rate": rounded(control_match_rate), "source_receiver_node_corr": rounded(safe_corr(source_vals, receiver_vals)), "claim_posture": posture, "smallest_supported_claim": "Residual path-proxy handle supported above controls for this prompt family." if supported else "Do not write a path claim; use node-effect or failed-control language.", "primary_caveat": caveat})
        for c in close_controls:
            counterexamples.append({"domain": domain, "item_id": best.get("item_id", ""), "kind": "control_matches_mediated_path", "path_id": best.get("path_id", ""), "source_depth": best.get("source_depth", ""), "receiver_depth": best.get("receiver_depth", ""), "path_recovery": best.get("mediated_path_recovery", ""), "control_recovery": c.get("control_recovery", ""), "control": c.get("control", ""), "lesson": "The path proxy does not beat this control by a comfortable margin."})
        best_node = safe_max([best.get("source_node_recovery"), best.get("receiver_node_recovery")])
        if math.isfinite(best_node) and fnum(best.get("mediated_path_recovery")) < best_node - CONTROL_CLOSE_TOL:
            counterexamples.append({"domain": domain, "item_id": best.get("item_id", ""), "kind": "node_effect_dominates_path_proxy", "path_id": best.get("path_id", ""), "source_depth": best.get("source_depth", ""), "receiver_depth": best.get("receiver_depth", ""), "path_recovery": best.get("mediated_path_recovery", ""), "control_recovery": rounded(best_node), "control": "best_node_patch", "lesson": "The strongest ordinary node patch is larger than the mediated receiver proxy."})
        if fnum(best.get("joint_increment_over_best_node")) <= 0:
            counterexamples.append({"domain": domain, "item_id": best.get("item_id", ""), "kind": "joint_patch_not_superadditive", "path_id": best.get("path_id", ""), "source_depth": best.get("source_depth", ""), "receiver_depth": best.get("receiver_depth", ""), "path_recovery": best.get("joint_clean_two_site_recovery", ""), "control_recovery": best.get("receiver_node_recovery", ""), "control": "max_node_effect", "lesson": "The clean two-site patch is not stronger than the best node patch; avoid interaction language."})
    counterexamples = counterexamples[:MAX_COUNTEREXAMPLES]
    metrics = {"n_tasks": len(tasks), "n_behavior_pass_tasks": sum(1 for t in tasks if t.baseline_pass), "n_node_rows": len(node_rows), "n_path_rows": len(path_rows), "n_control_rows": len(control_rows), "n_counterexamples": len(counterexamples), "claim_ready_domains": sum(1 for r in evidence if r.get("claim_posture") in {"path_proxy_supported", "path_proxy_plus_joint_interaction"}), "domains": [r["domain"] for r in evidence], "thresholds": {"baseline_margin": BASELINE_MARGIN, "min_denominator": MIN_DENOMINATOR, "min_mediated_recovery": MIN_MEDIATED_RECOVERY, "min_specificity_gap": MIN_SPECIFICITY_GAP, "control_close_tolerance": CONTROL_CLOSE_TOL}}
    return evidence, counterexamples, metrics


def write_task_manifest(ctx: bench.RunContext, tasks: Sequence[PathTask]) -> None:
    rows = [{"item_id": t.item_id, "domain": t.domain, "path_id": t.path_id, "clean_prompt": t.clean_prompt, "corrupt_prompt": t.corrupt_prompt, "target": bench.visible_token(t.target), "distractor": bench.visible_token(t.distractor), "source_pos": t.source_pos, "receiver_pos": t.receiver_pos, "final_pos": t.final_pos, "wrong_pos": t.wrong_pos, "candidate_nodes_json": json.dumps(t.candidate_nodes, sort_keys=True), "candidate_paths_json": json.dumps(t.candidate_paths, sort_keys=True), "baseline_pass": t.baseline_pass, "baseline_drop_reason": t.baseline_drop_reason} for t in tasks]
    path = ctx.path("tables", "task_manifest.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Selected Lab 27 tasks, runtime positions, and candidate path metadata.")


def write_tables(ctx: bench.RunContext, node_rows: Sequence[Mapping[str, Any]], path_rows: Sequence[Mapping[str, Any]], control_rows: Sequence[Mapping[str, Any]], accounting_rows: Sequence[Mapping[str, Any]], evidence: Sequence[Mapping[str, Any]], counterexamples: Sequence[Mapping[str, Any]], selection_rows: Sequence[Mapping[str, Any]]) -> None:
    specs = [
        ("tables/node_effect_baseline.csv", node_rows, "Node-level residual patch effects."),
        ("tables/depth_selection.csv", selection_rows, "Node-screen depths promoted into the path grid."),
        ("tables/path_patch_report.csv", path_rows, "Mediated path-proxy and clean two-site rows."),
        ("tables/path_specificity_controls.csv", control_rows, "Wrong-receiver, random-source, and reverse-site controls."),
        ("tables/mediation_accounting.csv", accounting_rows, "Source, receiver, mediated, joint, and control accounting."),
        ("tables/path_evidence_matrix.csv", evidence, "Domain-level path-proxy evidence matrix."),
        ("tables/path_counterexamples.csv", counterexamples, "Counterexamples that defeat or narrow path language."),
    ]
    for rel, rows, desc in specs:
        path = ctx.path(*rel.split("/"))
        bench.write_csv_with_context(ctx, path, rows)
        ctx.register_artifact(path, "table", desc)
    results = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results, path_rows)
    ctx.register_artifact(results, "table", "Alias of tables/path_patch_report.csv for dashboard tooling.")


def write_state(ctx: bench.RunContext, bundle: bench.ModelBundle, tasks: Sequence[PathTask], depths: Sequence[int], selection_rows: Sequence[Mapping[str, Any]]) -> None:
    payload = {"lab": "lab27", "model": bundle.anatomy.model_id, "n_layers": bundle.anatomy.n_layers, "stream_depth_convention": "streams[k] is pre-norm residual after k blocks; k=0 embeddings, k=L final-norm input", "depth_grid": list(map(int, depths)), "thresholds": {"baseline_margin": BASELINE_MARGIN, "min_denominator": MIN_DENOMINATOR, "min_mediated_recovery": MIN_MEDIATED_RECOVERY, "min_specificity_gap": MIN_SPECIFICITY_GAP, "control_close_tolerance": CONTROL_CLOSE_TOL}, "selected_depths": list(selection_rows), "tasks": [{"item_id": t.item_id, "domain": t.domain, "path_id": t.path_id, "positions": t.positions, "raw_positions": t.raw_positions, "token_offset": t.token_offset, "candidate_nodes": t.candidate_nodes, "candidate_paths": t.candidate_paths, "baseline_pass": t.baseline_pass, "baseline_drop_reason": t.baseline_drop_reason} for t in tasks]}
    path = ctx.path("state", "path_candidates.json")
    bench.write_json(path, payload)
    ctx.register_artifact(path, "state", "Task candidate sites, selected depths, and thresholds for Lab 27.")


def write_cards(ctx: bench.RunContext, bundle: bench.ModelBundle, data_info: Mapping[str, Any], metrics: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]], counterexamples: Sequence[Mapping[str, Any]]) -> None:
    verdict_lines = ["| domain | pass tasks | best item | src depth | recv depth | mediated | control floor | gap | posture |", "|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for r in evidence:
        verdict_lines.append(f"| {r['domain']} | {r['n_behavior_pass']} | `{r.get('best_item', '')}` | {r.get('source_depth', '')} | {r.get('receiver_depth', '')} | {r.get('best_mediated_path_recovery', '')} | {r.get('control_floor', '')} | {r.get('specificity_gap', '')} | {r.get('claim_posture', '')} |")
    method = ["# Lab 27 method card", "", "This run uses a residual source-to-receiver mediation proxy. It is not a full edge-level path-patching implementation.", "", f"- model: `{bundle.anatomy.model_id}` ({bundle.anatomy.n_layers} blocks)", f"- tasks: {metrics.get('n_tasks', 0)} selected; {metrics.get('n_behavior_pass_tasks', 0)} passed the behavior gate", "- intervention: clean-to-corrupt residual interchange at source sites, receiver sites, and source-patched receiver states", "- evidence rung: `CAUSAL`, scoped", "- non-claim: the proxy does not identify a unique attention head, MLP, or edge", "", "## Verdict table", "", *verdict_lines, "", "Safe sentence: `The residual source-to-receiver path proxy recovered behavior above controls on this prompt family.`", "", "Unsafe sentence: `This proves the exact internal edge from A to B in all contexts.`", ""]
    path = ctx.path("method_card.md")
    bench.write_text(path, "\n".join(method))
    ctx.register_artifact(path, "summary", "Method card and scoped path-proxy verdict.")

    passed = sum(1 for r in evidence if r.get("claim_posture") in {"path_proxy_supported", "path_proxy_plus_joint_interaction"})
    audit_result = "passed" if passed and passed == len(evidence) else ("mixed" if passed else "failed")
    audit = ["# Lab 27 operationalization audit", "", "```yaml", "headline_claim: \"a source-to-receiver route carries behavior beyond ordinary node effects\"", "cheap_explanation: \"receiver patching, wrong-site disruption, or random-source perturbations explain recovery\"", "killer_control: \"wrong-receiver, random-source, reverse-site, node-dominance, and behavior-gate checks\"", f"result: \"{audit_result}\"", f"claim_allowed: \"{'handle' if passed else 'no path claim'}\"", "```", "", "## What the proxy can say", "", "It can say that a receiver state produced by a clean source patch was behaviorally useful under this residual interchange battery.", "", "## What it cannot say", "", "It does not isolate attention Q/K/V reads, MLP inputs, or a unique computational edge. All routes from the patched source site to the receiver remain available in the source-patched run.", "", "## Domain verdicts", ""]
    audit += [f"- `{r['domain']}`: `{r['claim_posture']}`. Smallest claim: {r['smallest_supported_claim']} Caveat: {r['primary_caveat']}" for r in evidence]
    audit += ["", "## Counterexamples", ""]
    audit += [f"- `{r['domain']}` `{r['item_id']}` `{r['kind']}`: {r['lesson']}" for r in counterexamples] if counterexamples else ["- No automatic counterexample crossed the thresholds. Replicate before broadening the claim."]
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(audit) + "\n")
    ctx.register_artifact(path, "summary", "Operationalization audit for Lab 27 path-proxy claims.")

    science_ready = bool(data_info.get("science_ready", True)) and metrics.get("n_behavior_pass_tasks", 0) > 0
    strongest = next((r for r in evidence if r.get("claim_posture") in {"path_proxy_supported", "path_proxy_plus_joint_interaction"}), None)
    smallest = strongest.get("smallest_supported_claim") if strongest else "No path-proxy claim survived the automatic gates in this run."
    main_counter = counterexamples[0]["lesson"] if counterexamples else "No automatic counterexample crossed thresholds; inspect controls manually."
    summary = ["# Lab 27 run summary: path-specific patching and causal mediation", "", f"- model: `{bundle.anatomy.model_id}`", f"- data rows: {data_info['n_rows_selected']} selected from `{pathlib.Path(str(data_info['data_path'])).name}`", f"- domains: `{data_info['domains_selected']}`", f"- science_ready: `{str(science_ready).lower()}`", "- method: residual source-to-receiver mediation proxy, not exact edge isolation", f"- smallest surviving claim: {smallest}", f"- main counterexample: {main_counter}", "", "## Headline verdicts", "", *verdict_lines, "", "## Reading order", "", "1. `method_card.md` for claim boundaries.", "2. `tables/baseline_behavior.csv` for behavior gates.", "3. `tables/node_effect_baseline.csv` and `tables/depth_selection.csv` for node screens.", "4. `tables/path_patch_report.csv` and `tables/path_specificity_controls.csv` for path rows and controls.", "5. `tables/path_counterexamples.csv` and `operationalization_audit.md` before writing path language.", "", "## Caveats", "", "- The mediated proxy uses all routes between source and receiver that remain in the model after the source patch.", "- A strong receiver node patch is not a path claim.", "- Tier A is a smoke path and may be an honest negative.", "", f"Intervention rows: node={metrics['n_node_rows']}, path={metrics['n_path_rows']}, controls={metrics['n_control_rows']}.", ""]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(summary))
    ctx.register_artifact(path, "summary", "Run summary with verdicts, reading order, and surviving claim boundary.")


def write_claims(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]]) -> None:
    claims = []
    for i, r in enumerate(evidence, start=1):
        supported = r.get("claim_posture") in {"path_proxy_supported", "path_proxy_plus_joint_interaction"}
        claims.append({"id": f"{LAB_ID}-C{i}", "tag": "CAUSAL" if supported else "CAUSAL,AUDIT", "text": f"In `{r['domain']}`, Lab 27's residual source-to-receiver proxy at source depth {r.get('source_depth', '')} and receiver depth {r.get('receiver_depth', '')} had mediated recovery {r.get('best_mediated_path_recovery', '')} and specificity gap {r.get('specificity_gap', '')}. Posture: {r.get('claim_posture', '')}. This is a path-proxy handle, not a unique-edge claim.", "artifact": f"runs/{ctx.run_dir.name}/tables/path_evidence_matrix.csv", "falsifier": "Wrong-receiver, random-source, reverse-site, or node-only controls match the mediated recovery on held-out prompt families."})
    if not claims:
        claims.append({"id": f"{LAB_ID}-N1", "tag": "CAUSAL,AUDIT", "text": "No Lab 27 path-proxy row was produced; inspect behavior and tokenization gates before making path claims.", "artifact": f"runs/{ctx.run_dir.name}/run_summary.md", "falsifier": "A rerun with behavior-gated tasks produces path rows that clear the control gates."})
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)


def write_plot_guide(ctx: bench.RunContext) -> None:
    rows = [
        {"plot": "path_mediation_dashboard.png", "first_question": "Did any domain clear behavior, mediation, and control gates together?", "concept": "One-screen path proxy cockpit."},
        {"plot": "node_vs_path_effects.png", "first_question": "Are ordinary node effects enough?", "concept": "Best source/receiver node effects versus mediated path proxy."},
        {"plot": "path_specificity_matrix.png", "first_question": "Which cells beat controls?", "concept": "Source-depth by receiver-depth specificity gaps for the best domain."},
        {"plot": "mediation_accounting_waterfall.png", "first_question": "How do source, receiver, mediated, joint, and control terms compare?", "concept": "Best-cell accounting."},
        {"plot": "heldout_path_transfer.png", "first_question": "Is the story broad across domains?", "concept": "Domain-level mean and best specificity gaps."},
        {"plot": "path_graph.png", "first_question": "Where should stricter path patching look next?", "concept": "Schematic of best residual route proxies."},
    ]
    path = ctx.path("tables", "plot_reading_guide.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Map from Lab 27 plots to the concept and caveat each protects.")


def write_placeholder(ctx: bench.RunContext, name: str, title: str, message: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.40, message, ha="center", va="center", fontsize=10, wrap=True)
    bench.save_figure(ctx, fig, name, title)


def write_plots(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]], path_rows: Sequence[Mapping[str, Any]], control_rows: Sequence[Mapping[str, Any]]) -> None:
    write_plot_guide(ctx)
    if ctx.args.no_plots:
        return
    import matplotlib.pyplot as plt
    import numpy as np

    if not evidence:
        for name in ("path_mediation_dashboard.png", "node_vs_path_effects.png", "path_specificity_matrix.png", "mediation_accounting_waterfall.png", "heldout_path_transfer.png", "path_graph.png"):
            write_placeholder(ctx, name, name.replace("_", " ").replace(".png", ""), "No evidence rows were produced.")
        return
    labels = [str(r["domain"]) for r in evidence]
    x = np.arange(len(labels))
    mediated = [fnum(r.get("best_mediated_path_recovery"), 0.0) for r in evidence]
    controls = [fnum(r.get("control_floor"), 0.0) for r in evidence]
    source = [fnum(r.get("best_source_node_recovery"), 0.0) for r in evidence]
    receiver = [fnum(r.get("best_receiver_node_recovery"), 0.0) for r in evidence]
    joint = [fnum(r.get("best_joint_clean_two_site_recovery"), 0.0) for r in evidence]
    gaps = [fnum(r.get("specificity_gap"), 0.0) for r in evidence]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    fig.suptitle("Lab 27 path mediation dashboard", fontsize=14, fontweight="bold")
    axes[0, 0].bar(x - 0.18, mediated, 0.36, label="mediated path proxy")
    axes[0, 0].bar(x + 0.18, controls, 0.36, label="strongest control")
    axes[0, 0].axhline(MIN_MEDIATED_RECOVERY, linestyle="--", linewidth=1, label="recovery gate")
    axes[0, 0].set_xticks(x, labels, rotation=15, ha="right"); axes[0, 0].set_ylabel("recovery"); axes[0, 0].set_title("Best mediated path proxy vs controls"); axes[0, 0].legend()
    axes[0, 1].bar(x - 0.27, source, 0.27, label="source node")
    axes[0, 1].bar(x, receiver, 0.27, label="receiver node")
    axes[0, 1].bar(x + 0.27, joint, 0.27, label="clean two-site")
    axes[0, 1].set_xticks(x, labels, rotation=15, ha="right"); axes[0, 1].set_ylabel("recovery"); axes[0, 1].set_title("Node effects and joint patch"); axes[0, 1].legend(fontsize=8)
    axes[1, 0].bar(labels, gaps); axes[1, 0].axhline(MIN_SPECIFICITY_GAP, linestyle="--", linewidth=1, label="specificity gate"); axes[1, 0].axhline(0, linewidth=0.8); axes[1, 0].set_ylabel("gap"); axes[1, 0].set_title("Specificity gap by domain"); axes[1, 0].tick_params(axis="x", rotation=15); axes[1, 0].legend()
    counts: dict[str, int] = defaultdict(int)
    for row in control_rows:
        if row.get("control_matches_path"):
            counts[str(row.get("control"))] += 1
    axes[1, 1].bar(list(counts) or ["none"], list(counts.values()) or [0]); axes[1, 1].set_ylabel("cell count"); axes[1, 1].set_title("Controls close to path proxy"); axes[1, 1].tick_params(axis="x", rotation=20)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); bench.save_figure(ctx, fig, "path_mediation_dashboard.png", "Lab 27 path mediation dashboard.")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - 0.25, source, 0.25, label="source"); ax.bar(x, receiver, 0.25, label="receiver"); ax.bar(x + 0.25, mediated, 0.25, label="mediated path")
    ax.axhline(0, linewidth=0.8); ax.set_xticks(x, labels, rotation=15, ha="right"); ax.set_ylabel("recovery"); ax.set_title("Node effects versus mediated path proxy"); ax.legend(); fig.tight_layout(); bench.save_figure(ctx, fig, "node_vs_path_effects.png", "Node effects versus mediated path proxy.")

    best_domain = max(evidence, key=lambda r: fnum(r.get("specificity_gap"), 0.0)).get("domain")
    rows = [r for r in path_rows if r.get("domain") == best_domain]
    if rows:
        srcs = sorted({int(r["source_depth"]) for r in rows}); recvs = sorted({int(r["receiver_depth"]) for r in rows})
        mat = np.full((len(recvs), len(srcs)), np.nan)
        for r in rows:
            mat[recvs.index(int(r["receiver_depth"])), srcs.index(int(r["source_depth"]))] = fnum(r.get("specificity_gap"))
        scale = max(0.2, float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 0.2)
        fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(srcs) + 3), max(4.8, 0.45 * len(recvs) + 2)))
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-scale, vmax=scale)
        ax.set_xticks(range(len(srcs)), srcs); ax.set_yticks(range(len(recvs)), recvs)
        ax.set_xlabel("source depth"); ax.set_ylabel("receiver depth"); ax.set_title(f"Specificity gap matrix: {best_domain}")
        for i in range(len(recvs)):
            for j in range(len(srcs)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.035, label="gap"); fig.tight_layout(); bench.save_figure(ctx, fig, "path_specificity_matrix.png", "Selected path-cell specificity gaps.")
    else:
        write_placeholder(ctx, "path_specificity_matrix.png", "Path specificity matrix", "No path rows were produced.")

    best = max(evidence, key=lambda r: fnum(r.get("specificity_gap"), 0.0))
    names = ["source", "receiver", "mediated", "joint", "control", "gap"]
    vals = [fnum(best.get("best_source_node_recovery"), 0), fnum(best.get("best_receiver_node_recovery"), 0), fnum(best.get("best_mediated_path_recovery"), 0), fnum(best.get("best_joint_clean_two_site_recovery"), 0), fnum(best.get("control_floor"), 0), fnum(best.get("specificity_gap"), 0)]
    fig, ax = plt.subplots(figsize=(9, 4.8)); ax.bar(names, vals); ax.axhline(0, linewidth=0.8); ax.axhline(MIN_SPECIFICITY_GAP, linestyle="--", linewidth=1, label="specificity gate"); ax.set_ylabel("recovery / gap"); ax.set_title(f"Best-cell accounting: {best['domain']} / {best.get('best_item', '')}"); ax.tick_params(axis="x", rotation=20); ax.legend(); fig.tight_layout(); bench.save_figure(ctx, fig, "mediation_accounting_waterfall.png", "Best-cell mediation accounting.")

    mean_gaps = [fnum(r.get("mean_specificity_gap"), 0) for r in evidence]
    fig, ax = plt.subplots(figsize=(8.5, 4.8)); ax.plot(x, mean_gaps, marker="o", label="mean cell gap"); ax.plot(x, gaps, marker="o", label="best cell gap"); ax.axhline(MIN_SPECIFICITY_GAP, linestyle="--", linewidth=1, label="specificity gate"); ax.axhline(0, linewidth=0.8); ax.set_xticks(x, labels, rotation=15, ha="right"); ax.set_ylabel("specificity gap"); ax.set_title("Domain breadth check"); ax.legend(); fig.tight_layout(); bench.save_figure(ctx, fig, "heldout_path_transfer.png", "Domain-level mean and best path-proxy gaps.")

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.65 * len(evidence) + 2))); ax.axis("off")
    for y, row in zip(range(len(evidence))[::-1], sorted(evidence, key=lambda r: fnum(r.get("specificity_gap"), 0), reverse=True)):
        ax.scatter([0.18, 0.78], [y, y], s=[180, 180]); ax.annotate("", xy=(0.74, y), xytext=(0.22, y), arrowprops={"arrowstyle": "->", "lw": 1.6}); ax.text(0.18, y + 0.16, f"source\nd{row.get('source_depth', '')}", ha="center", fontsize=8); ax.text(0.78, y + 0.16, f"receiver\nd{row.get('receiver_depth', '')}", ha="center", fontsize=8); ax.text(0.48, y + 0.12, f"{row['domain']} · gap {row.get('specificity_gap', '')}", ha="center", fontsize=9); ax.text(0.48, y - 0.18, str(row.get("claim_posture", "")).replace("_", " "), ha="center", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(-0.8, len(evidence) - 0.2); ax.set_title("Best residual route proxies by domain"); fig.tight_layout(); bench.save_figure(ctx, fig, "path_graph.png", "Compact route schematic for best path-proxy cells.")


def write_status_files(ctx: bench.RunContext, data_info: Mapping[str, Any], hook_check: Mapping[str, Any], lens_check: Mapping[str, Any], patch_noop: Mapping[str, Any], token_rows: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> None:
    science_ready = bool(data_info.get("science_ready", True)) and metrics.get("n_behavior_pass_tasks", 0) > 0
    safety = {"lab": "lab27", "unsafe_prompt_sampling": False, "refusal_ablation": False, "harmful_completion_generation": False, "generated_text_scoring": False, "blocked_rows": 0, "public_private_boundary_relevant": False, "science_ready": science_ready, "note": "Forward-pass-only residual patching on benign completion prompts."}
    path = ctx.path("diagnostics", "safety_status.json"); bench.write_json(path, safety); ctx.register_artifact(path, "diagnostic", "Safety and scope status for Lab 27.")
    checks = {"hook_parity_ok": bool(hook_check.get("ok")), "lens_self_check_ok": bool(lens_check.get("ok")), "patch_noop_ok": bool(patch_noop.get("ok")), "tokenization_kept": sum(1 for r in token_rows if r.get("kept")), "tokenization_dropped": sum(1 for r in token_rows if not r.get("kept")), "behavior_pass_tasks": metrics.get("n_behavior_pass_tasks", 0), "path_rows": metrics.get("n_path_rows", 0), "control_rows": metrics.get("n_control_rows", 0), "ok_for_science": bool(hook_check.get("ok")) and bool(lens_check.get("ok")) and bool(patch_noop.get("ok")) and science_ready}
    path = ctx.path("diagnostics", "self_check_status.json"); bench.write_json(path, checks); ctx.register_artifact(path, "diagnostic", "Aggregated self-check status for Lab 27.")


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    tasks, data_info = load_tasks(ctx)
    manifest = ctx.path("diagnostics", "data_manifest.json")
    bench.write_json(manifest, data_info)
    ctx.register_artifact(manifest, "diagnostic", "Lab 27 data manifest, hash, and science-ready status.")
    hook_check = bench.run_hook_parity_check(ctx, bundle, tasks[0].clean_prompt)
    first = bench.run_with_residual_cache(bundle, tasks[0].clean_prompt)
    lens_check = bench.run_lens_self_check(ctx, bundle, first)
    patch_noop = bench.run_patch_noop_check(ctx, bundle, tasks[0].clean_prompt)
    tasks, token_rows = tokenization_gate(ctx, bundle, tasks)
    clean_caps, _corrupt_caps, _baseline_rows = cache_baselines(ctx, bundle, tasks)
    write_task_manifest(ctx, tasks)
    depths = depth_grid(bundle.anatomy.n_layers, str(ctx.args.prompt_set))
    node_rows = run_node_screen(ctx, bundle, tasks, clean_caps, depths)
    path_rows, control_rows, accounting_rows, selection_rows = run_path_interventions(ctx, bundle, tasks, clean_caps, node_rows, depths)
    evidence, counterexamples, metrics = summarize(tasks, node_rows, path_rows, control_rows)
    metrics = {**metrics, "data": data_info, "depth_grid": list(map(int, depths))}
    write_tables(ctx, node_rows, path_rows, control_rows, accounting_rows, evidence, counterexamples, selection_rows)
    write_state(ctx, bundle, tasks, depths, selection_rows)
    write_status_files(ctx, data_info, hook_check, lens_check, patch_noop, token_rows, metrics)
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 27 metrics, gates, and thresholds.")
    write_cards(ctx, bundle, data_info, metrics, evidence, counterexamples)
    write_claims(ctx, evidence)
    write_plots(ctx, evidence, path_rows, control_rows)
    print(f"[lab27] wrote {len(evidence)} evidence rows and {len(counterexamples)} counterexamples")
