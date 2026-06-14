"""Lab 29: Training dynamics and circuit birth.

This lab is a controlled time-lapse experiment. It trains a tiny causal
transformer on an induction-copy task during the run, freezes several
checkpoints, and asks when different measurements become visible:
behavior, linear decodability, attention motifs, feature-lineage stability,
and a final-direction intervention-transfer check.

The code is intentionally boring where the science is fragile. Every event is
a threshold crossing with controls; none of the artifacts claim an exact moment
when the model "learned" a concept.
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

LAB_ID = "L29"
DATA_FILE = "training_dynamics_tasks.csv"
REQUIRED_COLUMNS = {
    "item_id", "task_family", "prompt", "target", "distractor", "split",
    "expected_mechanism", "notes",
}

INDUCTION_TOKENS = (
    "red", "blue", "green", "yellow", "orange", "purple", "silver",
    "bronze", "cyan", "magenta", "amber", "teal", "violet",
)
CONTROL_TOKENS = (
    "capital", "France", "Paris", "Rome", "language", "Italy", "Italian",
    "German", "month", "after", "before", "January", "February", "March",
    "April", "known", "as", "country", "calendar", "next",
)

PROMPT_SET_CAPS = {"small": 11, "medium": 18, "full": 0}
TRAIN_STEPS_BY_PROMPT_SET = {"small": 80, "medium": 220, "full": 420}
CHECKPOINT_FRACTIONS = (0.0, 0.04, 0.10, 0.25, 0.50, 0.75, 1.0)
BATCH_BY_PROMPT_SET = {"small": 64, "medium": 128, "full": 192}
PROBE_SIZES_BY_PROMPT_SET = {
    "small": (52, 26),
    "medium": (130, 78),
    "full": (260, 156),
}

TINY_D_MODEL = 64
TINY_LAYERS = 2
TINY_HEADS = 4
TINY_MAX_LEN = 8
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
INTERVENTION_NORM_FRACTION = 0.75

BEHAVIOR_ACC_BAR = 0.75
PROBE_ACC_BAR = 0.75
PROBE_SELECTIVITY_BAR = 0.20
MOTIF_GAP_BAR = 0.12
INTERVENTION_GAP_BAR = 0.20
CONTROL_LEAKAGE_ACC_BAR = 0.60
CLOSE_CONTROL_TOL = 0.05


@dataclasses.dataclass(frozen=True)
class Lab29Config:
    prompt_set: str
    train_steps: int
    checkpoint_steps: list[int]
    batch_size: int
    probe_train: int
    probe_eval: int
    seed: int


@dataclasses.dataclass
class DynamicsTask:
    item_id: str
    task_family: str
    prompt: str
    target: str
    distractor: str
    split: str
    expected_mechanism: str
    notes: str
    prompt_ids: list[int] = dataclasses.field(default_factory=list)
    target_id: int = -1
    distractor_id: int = -1
    kept: bool = False
    drop_reason: str = ""

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "DynamicsTask":
        return cls(
            item_id=str(row.get("item_id", "")).strip(),
            task_family=str(row.get("task_family", "")).strip(),
            prompt=str(row.get("prompt", "")).strip(),
            target=str(row.get("target", "")).strip(),
            distractor=str(row.get("distractor", "")).strip(),
            split=str(row.get("split", "")).strip() or "eval",
            expected_mechanism=str(row.get("expected_mechanism", "")).strip(),
            notes=str(row.get("notes", "")).strip(),
        )


@dataclasses.dataclass(frozen=True)
class ProbeExample:
    prompt_ids: list[int]
    target_id: int
    split: str


# ---------------------------------------------------------------------------
# Small utilities
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


def safe_median(values: Sequence[Any], default: float = float("nan")) -> float:
    vals = [fnum(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(statistics.median(vals)) if vals else default


def bool_mean(rows: Sequence[Mapping[str, Any]], key: str, default: float = float("nan")) -> float:
    vals = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            vals.append(1.0 if value else 0.0)
        elif value in {"True", "true", "1", 1}:
            vals.append(1.0)
        elif value in {"False", "false", "0", 0}:
            vals.append(0.0)
    return safe_mean(vals, default)


def row_float(row: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    return fnum(row.get(key), default)


def configure_torch_runtime() -> None:
    """Keep the tiny CPU smoke path fast and deterministic.

    PyTorch can spend more time coordinating many CPU threads than doing the
    tiny matrix multiplications in this lab. One thread is faster for the
    in-course transformer and avoids smoke-test timeouts on shared VMs.
    """
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


def data_path(args: Any) -> pathlib.Path:
    prompt_set = str(getattr(args, "prompt_set", "") or "")
    candidate = pathlib.Path(prompt_set)
    if prompt_set not in PROMPT_SET_CAPS and candidate.suffix.lower() in {".csv", ".tsv"}:
        return candidate if candidate.is_absolute() else (bench.COURSE_ROOT / candidate).resolve()
    return bench.COURSE_ROOT / "data" / DATA_FILE


def build_config(ctx: bench.RunContext) -> Lab29Config:
    prompt_set = str(ctx.args.prompt_set)
    if prompt_set not in PROMPT_SET_CAPS:
        prompt_set = "small"
    steps = TRAIN_STEPS_BY_PROMPT_SET.get(prompt_set, TRAIN_STEPS_BY_PROMPT_SET["small"])
    checkpoint_steps = sorted({int(round(steps * frac)) for frac in CHECKPOINT_FRACTIONS} | {steps})
    probe_train, probe_eval = PROBE_SIZES_BY_PROMPT_SET.get(prompt_set, PROBE_SIZES_BY_PROMPT_SET["small"])
    return Lab29Config(
        prompt_set=prompt_set,
        train_steps=steps,
        checkpoint_steps=checkpoint_steps,
        batch_size=BATCH_BY_PROMPT_SET.get(prompt_set, BATCH_BY_PROMPT_SET["small"]),
        probe_train=probe_train,
        probe_eval=probe_eval,
        seed=int(ctx.args.seed),
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def builtin_rows() -> list[dict[str, str]]:
    """Small deterministic fallback. It is smoke-only unless the CSV is committed."""
    rows: list[dict[str, str]] = []

    def add(item_id: str, family: str, prompt: str, target: str, distractor: str, split: str, mechanism: str, notes: str) -> None:
        rows.append({
            "item_id": item_id,
            "task_family": family,
            "prompt": prompt,
            "target": target,
            "distractor": distractor,
            "split": split,
            "expected_mechanism": mechanism,
            "notes": notes,
        })

    induction_specs = [
        ("ind_train_01", "red blue green red blue", "green", "yellow", "train"),
        ("ind_train_02", "orange purple cyan orange purple", "cyan", "red", "train"),
        ("ind_train_03", "silver bronze teal silver bronze", "teal", "orange", "train"),
        ("ind_train_04", "magenta amber violet magenta amber", "violet", "blue", "train"),
        ("ind_heldout_01", "yellow green purple yellow green", "purple", "amber", "heldout"),
        ("ind_heldout_02", "cyan red orange cyan red", "orange", "silver", "heldout"),
        ("ind_test_01", "teal violet magenta teal violet", "magenta", "green", "test"),
        ("ind_test_02", "bronze yellow silver bronze yellow", "silver", "purple", "test"),
    ]
    for item_id, prompt, target, distractor, split in induction_specs:
        add(item_id, "induction_copy", prompt, target, distractor, split, "previous_match_copy", "trained synthetic induction-copy row")
    add("ctrl_relation_01", "relation_control", "capital France", "Paris", "Rome", "control", "not_trained", "untrained relation-control row")
    add("ctrl_relation_02", "relation_control", "language Italy", "Italian", "German", "control", "not_trained", "untrained relation-control row")
    add("ctrl_calendar_01", "calendar_control", "month after January", "February", "March", "control", "not_trained", "untrained calendar-control row")
    add("ctrl_calendar_02", "calendar_control", "month before March", "February", "April", "control", "not_trained", "untrained calendar-control row")
    return rows


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


def balanced_cap(tasks: Sequence[DynamicsTask], cap: int) -> list[DynamicsTask]:
    if cap <= 0 or len(tasks) <= cap:
        return list(tasks)
    by_family: dict[str, list[DynamicsTask]] = defaultdict(list)
    for task in tasks:
        by_family[task.task_family].append(task)
    out: list[DynamicsTask] = []
    cursor = 0
    families = sorted(by_family)
    while len(out) < cap:
        made_progress = False
        for family in families:
            if cursor < len(by_family[family]):
                out.append(by_family[family][cursor])
                made_progress = True
                if len(out) >= cap:
                    break
        if not made_progress:
            break
        cursor += 1
    return out


def load_tasks(ctx: bench.RunContext) -> tuple[list[DynamicsTask], dict[str, Any]]:
    path = data_path(ctx.args)
    expected_sha, manifest_note = manifest_expected_hash(path)
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            rows = [dict(row) for row in csv.DictReader(f)]
        if not rows:
            raise ValueError(f"{path} contains no rows")
        missing = sorted(REQUIRED_COLUMNS - set(rows[0]))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        actual_sha = file_sha256(path)
        data_source = "frozen_csv"
        science_ready = True
        fallback_warning = False
    else:
        rows = builtin_rows()
        actual_sha = hashlib.sha256("\n".join(r["item_id"] for r in rows).encode("utf-8")).hexdigest()
        data_source = "builtin_smoke_fallback"
        science_ready = False
        fallback_warning = True
        manifest_note = f"{path} not found; builtin smoke fallback used"
        print("[lab29] frozen CSV missing; using builtin smoke fallback. Do not ledger broad science claims from this run.")

    tasks = [DynamicsTask.from_row(row) for row in rows]
    bad = [t.item_id or "(blank)" for t in tasks if not t.item_id or not t.prompt or not t.target or not t.distractor]
    if bad:
        raise ValueError(f"{DATA_FILE} contains incomplete rows: {bad[:8]}")
    cap = PROMPT_SET_CAPS.get(str(ctx.args.prompt_set), 0)
    selected = balanced_cap(tasks, cap)
    if int(ctx.args.max_examples or 0) > 0:
        selected = balanced_cap(selected, int(ctx.args.max_examples))
    info = {
        "data_file": DATA_FILE,
        "data_path": str(path),
        "data_source": data_source,
        "data_sha256": actual_sha,
        "manifest_expected_sha256": expected_sha,
        "manifest_note": manifest_note,
        "manifest_ok": (actual_sha == expected_sha) if expected_sha else None,
        "n_rows_file": len(rows),
        "n_rows_selected": len(selected),
        "families": dict(Counter(t.task_family for t in selected)),
        "splits": dict(Counter(t.split for t in selected)),
        "science_ready": science_ready,
        "fallback_warning": fallback_warning,
        "science_scope": "controlled tiny-transformer training sequence; not an external pretrained-checkpoint claim",
    }
    return selected, info


def build_vocab(tasks: Sequence[DynamicsTask]) -> tuple[dict[str, int], list[str]]:
    tokens = ["<pad>"] + list(INDUCTION_TOKENS) + list(CONTROL_TOKENS)
    for task in tasks:
        tokens.extend(task.prompt.split())
        tokens.append(task.target)
        tokens.append(task.distractor)
    seen: set[str] = set()
    vocab: list[str] = []
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            vocab.append(tok)
    return {tok: i for i, tok in enumerate(vocab)}, vocab


def encode_prompt(stoi: Mapping[str, int], prompt: str) -> list[int]:
    return [stoi[tok] for tok in prompt.split()]


def tokenization_gate(ctx: bench.RunContext, tasks: list[DynamicsTask], stoi: Mapping[str, int], vocab: Sequence[str]) -> list[DynamicsTask]:
    rows: list[dict[str, Any]] = []
    kept: list[DynamicsTask] = []
    for task in tasks:
        problems: list[str] = []
        try:
            prompt_ids = encode_prompt(stoi, task.prompt)
            target_id = stoi[task.target]
            distractor_id = stoi[task.distractor]
        except KeyError as exc:
            problems.append(f"missing_vocab={exc}")
            prompt_ids = []
            target_id = -1
            distractor_id = -1
        if not prompt_ids:
            problems.append("empty_prompt")
        if len(prompt_ids) > TINY_MAX_LEN:
            problems.append(f"prompt_too_long>{TINY_MAX_LEN}")
        if target_id == distractor_id and target_id >= 0:
            problems.append("target_equals_distractor")
        if task.task_family == "induction_copy" and previous_match_position(prompt_ids) is None:
            problems.append("induction_row_has_no_previous_match")
        if not problems:
            task.prompt_ids = prompt_ids
            task.target_id = target_id
            task.distractor_id = distractor_id
            task.kept = True
            kept.append(task)
        else:
            task.drop_reason = ";".join(problems)
        rows.append({
            "item_id": task.item_id,
            "task_family": task.task_family,
            "split": task.split,
            "prompt": task.prompt,
            "prompt_ids": " ".join(str(i) for i in prompt_ids),
            "prompt_tokens": " ".join(vocab[i] for i in prompt_ids) if prompt_ids else "",
            "prompt_len": len(prompt_ids),
            "target": task.target,
            "target_id": target_id if target_id >= 0 else "",
            "distractor": task.distractor,
            "distractor_id": distractor_id if distractor_id >= 0 else "",
            "previous_match_position": previous_match_position(prompt_ids),
            "kept": not problems,
            "problems": ";".join(problems),
        })
    path = ctx.path("diagnostics", "tokenization_gate.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic", "Tiny-tokenizer vocabulary, prompt length, and induction-position audit.")
    if not kept:
        raise RuntimeError("Lab 29 tokenization gate dropped every task.")
    return kept


# ---------------------------------------------------------------------------
# Tiny model
# ---------------------------------------------------------------------------


def make_causal_mask(seq_len: int, device: Any) -> Any:
    import torch

    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


def build_tiny_classes() -> tuple[type, type]:
    import torch
    import torch.nn as nn

    class TinyBlock(nn.Module):
        def __init__(self, d_model: int, n_heads: int) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )

        def forward(self, x: Any, *, return_attn: bool = False) -> tuple[Any, Any | None]:
            y = self.ln1(x)
            mask = make_causal_mask(y.shape[1], y.device)
            attn_out, attn_weights = self.attn(
                y, y, y,
                attn_mask=mask,
                need_weights=return_attn,
                average_attn_weights=False,
            )
            x = x + attn_out
            x = x + self.mlp(self.ln2(x))
            return x, attn_weights if return_attn else None

    class TinyTransformer(nn.Module):
        def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, max_len: int) -> None:
            super().__init__()
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.n_layers = n_layers
            self.max_len = max_len
            self.token_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_len, d_model)
            self.blocks = nn.ModuleList([TinyBlock(d_model, n_heads) for _ in range(n_layers)])
            self.final_ln = nn.LayerNorm(d_model)
            self.unembed = nn.Linear(d_model, vocab_size, bias=False)

        def forward(
            self,
            input_ids: Any,
            *,
            return_cache: bool = False,
            addition: tuple[int, Any, float] | None = None,
        ) -> tuple[Any, list[Any], list[Any]]:
            batch, seq = input_ids.shape
            if seq > self.max_len:
                raise ValueError(f"sequence length {seq} exceeds max_len {self.max_len}")
            pos = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(batch, seq)
            x = self.token_emb(input_ids) + self.pos_emb(pos)
            streams: list[Any] = []
            attentions: list[Any] = []

            def maybe_add(depth: int, hidden: Any) -> Any:
                if addition is None or int(addition[0]) != depth:
                    return hidden
                vector, scale = addition[1], float(addition[2])
                hidden = hidden.clone()
                hidden[:, -1, :] = hidden[:, -1, :] + scale * vector.to(hidden.device, hidden.dtype)
                return hidden

            x = maybe_add(0, x)
            if return_cache:
                streams.append(x.detach().float().cpu())
            for i, block in enumerate(self.blocks, start=1):
                x, attn = block(x, return_attn=return_cache)
                x = maybe_add(i, x)
                if return_cache:
                    streams.append(x.detach().float().cpu())
                    attentions.append(attn.detach().float().cpu() if attn is not None else None)
            logits = self.unembed(self.final_ln(x))
            return logits, streams, attentions

    return TinyBlock, TinyTransformer


def clone_state_dict(model: Any) -> dict[str, Any]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def make_induction_batch(stoi: Mapping[str, int], batch_size: int, generator: Any, device: Any) -> tuple[Any, Any]:
    import torch

    color_ids = [stoi[tok] for tok in INDUCTION_TOKENS]
    rows: list[list[int]] = []
    labels: list[int] = []
    for _ in range(batch_size):
        perm = torch.randperm(len(color_ids), generator=generator)
        a, b, c = [color_ids[int(i)] for i in perm[:3]]
        rows.append([a, b, c, a, b])
        labels.append(c)
    return torch.tensor(rows, dtype=torch.long, device=device), torch.tensor(labels, dtype=torch.long, device=device)


def train_tiny_sequence(
    ctx: bench.RunContext,
    cfg: Lab29Config,
    stoi: Mapping[str, int],
    vocab: Sequence[str],
) -> tuple[Any, dict[int, dict[str, Any]], list[dict[str, Any]], Any]:
    import torch
    import torch.nn.functional as F

    _, TinyTransformerClass = build_tiny_classes()
    device = torch.device("cuda" if torch.cuda.is_available() and str(ctx.args.device) != "cpu" else "cpu")
    torch.manual_seed(cfg.seed)
    model = TinyTransformerClass(len(vocab), TINY_D_MODEL, TINY_LAYERS, TINY_HEADS, TINY_MAX_LEN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 29029)
    snapshots: dict[int, dict[str, Any]] = {0: clone_state_dict(model)}
    rows: list[dict[str, Any]] = [{"checkpoint_step": 0, "loss": "", "train_accuracy": "", "learning_rate": LEARNING_RATE}]
    report_every = max(1, cfg.train_steps // 5)

    for step in range(1, cfg.train_steps + 1):
        model.train()
        x, y = make_induction_batch(stoi, cfg.batch_size, gen, device)
        logits, _, _ = model(x)
        loss = F.cross_entropy(logits[:, -1, :], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in cfg.checkpoint_steps or step % report_every == 0:
            with torch.no_grad():
                pred = logits[:, -1, :].argmax(dim=-1)
                acc = float((pred == y).float().mean().detach().cpu())
            rows.append({
                "checkpoint_step": step,
                "loss": rounded(float(loss.detach().cpu())),
                "train_accuracy": rounded(acc),
                "learning_rate": LEARNING_RATE,
            })
            print(f"[lab29] tiny training step {step}/{cfg.train_steps} loss={float(loss.detach().cpu()):.4f} acc={acc:.3f}")
        if step in cfg.checkpoint_steps:
            snapshots[step] = clone_state_dict(model)

    path = ctx.path("tables", "tiny_training_log.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Tiny transformer training loss and batch accuracy log.")
    return model, snapshots, rows, device


def run_model_on_prompt(model: Any, ids: Sequence[int], device: Any, *, return_cache: bool = True, addition: tuple[int, Any, float] | None = None) -> tuple[Any, list[Any], list[Any]]:
    import torch

    x = torch.tensor([list(ids)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, streams, attentions = model(x, return_cache=return_cache, addition=addition)
    return logits[0, -1, :].detach().float().cpu(), streams, attentions


def logit_margin(logits: Any, target_id: int, distractor_id: int) -> float:
    return float(logits[target_id] - logits[distractor_id])


def previous_match_position(ids: Sequence[int]) -> int | None:
    if len(ids) < 2:
        return None
    query = ids[-1]
    for pos in range(len(ids) - 2, -1, -1):
        if ids[pos] == query:
            return pos
    return None


def top_tokens(logits: Any, vocab: Sequence[str], k: int = 3) -> str:
    import torch

    vals, idx = torch.topk(logits, k=min(k, len(vocab)))
    return " | ".join(f"{vocab[int(i)]}:{float(v):.3f}" for v, i in zip(vals, idx))


def logit_lens_event_depth(model: Any, streams: Sequence[Any], target_id: int, distractor_id: int) -> tuple[int | None, list[float]]:
    margins: list[float] = []
    import torch

    for stream in streams:
        h = stream[0, -1, :].to(next(model.parameters()).device)
        with torch.no_grad():
            logits = model.unembed(model.final_ln(h))
        margins.append(float((logits[target_id] - logits[distractor_id]).detach().cpu()))
    event = next((i for i, margin in enumerate(margins) if margin > 0.0), None)
    return event, margins


def motif_scores(attentions: Sequence[Any], ids: Sequence[int]) -> dict[str, Any]:
    """Measure the induction source token, not merely the previous match.

    For prompts of the form ``A B C A B -> C``, the behaviorally useful token
    at the final position is the successor of the previous ``B`` occurrence
    (``C``), not the previous ``B`` itself. This distinction is the whole
    little hinge of the lab: a previous-token heatmap can look plausible while
    measuring the wrong motif.
    """
    prev = previous_match_position(ids)
    final_pos = len(ids) - 1
    source = (prev + 1) if prev is not None and (prev + 1) < final_pos else None
    if source is None or not attentions:
        return {
            "previous_match_position": prev,
            "induction_source_position": source,
            "best_induction_source_attention": float("nan"),
            "mean_induction_source_attention": float("nan"),
            "best_motif_layer": None,
            "best_motif_head": None,
            "random_attention_baseline": 1.0 / max(1, len(ids)),
        }
    all_scores: list[float] = []
    best_score = -float("inf")
    best_layer: int | None = None
    best_head: int | None = None
    for layer, attn in enumerate(attentions, start=1):
        if attn is None:
            continue
        # [batch, heads, target_position, source_position]
        scores = attn[0, :, final_pos, source]
        for head, score in enumerate(scores):
            val = float(score)
            all_scores.append(val)
            if val > best_score:
                best_score = val
                best_layer = layer
                best_head = head
    return {
        "previous_match_position": prev,
        "induction_source_position": source,
        "best_induction_source_attention": best_score,
        "mean_induction_source_attention": safe_mean(all_scores),
        "best_motif_layer": best_layer,
        "best_motif_head": best_head,
        "random_attention_baseline": 1.0 / max(1, len(ids)),
    }


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def evaluate_checkpoints(
    ctx: bench.RunContext,
    model: Any,
    snapshots: Mapping[int, Mapping[str, Any]],
    tasks: Sequence[DynamicsTask],
    vocab: Sequence[str],
    device: Any,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    caches: dict[int, dict[str, Any]] = {}
    for step in sorted(snapshots):
        model.load_state_dict(snapshots[step])
        model.eval()
        task_cache: dict[str, Any] = {}
        for task in tasks:
            logits, streams, attentions = run_model_on_prompt(model, task.prompt_ids, device)
            margin = logit_margin(logits, task.target_id, task.distractor_id)
            lens_event, lens_margins = logit_lens_event_depth(model, streams, task.target_id, task.distractor_id)
            motif = motif_scores(attentions, task.prompt_ids)
            row = {
                "checkpoint_step": step,
                "item_id": task.item_id,
                "task_family": task.task_family,
                "split": task.split,
                "trained_family": task.task_family == "induction_copy",
                "target": task.target,
                "distractor": task.distractor,
                "target_minus_distractor": rounded(margin),
                "correct": margin > 0.0,
                "top_token": vocab[int(logits.argmax())],
                "top_tokens": top_tokens(logits, vocab),
                "logit_lens_event_depth": "" if lens_event is None else lens_event,
                "logit_lens_margins_json": json.dumps([rounded(v) for v in lens_margins]),
                "final_depth_lens_margin": rounded(lens_margins[-1] if lens_margins else float("nan")),
                "previous_match_position": "" if motif["previous_match_position"] is None else motif["previous_match_position"],
                "induction_source_position": "" if motif["induction_source_position"] is None else motif["induction_source_position"],
                "best_induction_source_attention": rounded(motif["best_induction_source_attention"]),
                "mean_induction_source_attention": rounded(motif["mean_induction_source_attention"]),
                "best_prev_match_attention": rounded(motif["best_induction_source_attention"]),
                "mean_prev_match_attention": rounded(motif["mean_induction_source_attention"]),
                "random_attention_baseline": rounded(motif["random_attention_baseline"]),
                "induction_source_attention_gap": rounded(fnum(motif["mean_induction_source_attention"]) - fnum(motif["random_attention_baseline"])),
                "prev_match_attention_gap": rounded(fnum(motif["mean_induction_source_attention"]) - fnum(motif["random_attention_baseline"])),
                "best_motif_layer": "" if motif["best_motif_layer"] is None else motif["best_motif_layer"],
                "best_motif_head": "" if motif["best_motif_head"] is None else motif["best_motif_head"],
            }
            rows.append(row)
            task_cache[task.item_id] = {"logits": logits, "streams": streams, "attentions": attentions, "row": row}
        caches[step] = task_cache
    path = ctx.path("tables", "checkpoint_behavior.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Checkpoint behavior, logit-lens, and attention-motif measurements.")
    return rows, caches


def generate_probe_examples(stoi: Mapping[str, int], *, n_train: int, n_eval: int, seed: int) -> list[ProbeExample]:
    import torch

    gen = torch.Generator(device="cpu").manual_seed(seed + 292929)
    color_ids = [stoi[tok] for tok in INDUCTION_TOKENS]
    examples: list[ProbeExample] = []
    for split, n in (("probe_train", n_train), ("probe_eval", n_eval)):
        for i in range(n):
            c = color_ids[i % len(color_ids)]
            others = [x for x in color_ids if x != c]
            perm = torch.randperm(len(others), generator=gen)
            a, b = [others[int(j)] for j in perm[:2]]
            examples.append(ProbeExample(prompt_ids=[a, b, c, a, b], target_id=c, split=split))
    return examples


def collect_hidden_by_depth(model: Any, examples: Sequence[ProbeExample], device: Any) -> dict[int, tuple[Any, list[int], list[str]]]:
    import torch

    by_depth: dict[int, list[Any]] = defaultdict(list)
    labels: list[int] = []
    splits: list[str] = []
    for ex in examples:
        _, streams, _ = run_model_on_prompt(model, ex.prompt_ids, device)
        for depth, stream in enumerate(streams):
            by_depth[depth].append(stream[0, -1, :])
        labels.append(ex.target_id)
        splits.append(ex.split)
    return {depth: (torch.stack(rows, dim=0), labels, splits) for depth, rows in by_depth.items()}


def centroid_accuracy(hidden: Any, labels: Sequence[int], splits: Sequence[str], train_labels_override: Sequence[int] | None = None) -> tuple[float, dict[int, Any], int, int]:
    import torch
    import torch.nn.functional as F

    train_idx = [i for i, split in enumerate(splits) if split == "probe_train"]
    eval_idx = [i for i, split in enumerate(splits) if split == "probe_eval"]
    train_labels = list(train_labels_override) if train_labels_override is not None else [labels[i] for i in train_idx]
    classes = sorted(set(train_labels))
    centroids: dict[int, Any] = {}
    for cls in classes:
        idx = [train_idx[i] for i, lab in enumerate(train_labels) if lab == cls]
        if idx:
            centroids[cls] = hidden[idx].mean(dim=0)
    if len(centroids) < 2 or not eval_idx:
        return float("nan"), centroids, len(train_idx), len(eval_idx)
    used = sorted(centroids)
    eval_keep = [i for i in eval_idx if labels[i] in used]
    if not eval_keep:
        return float("nan"), centroids, len(train_idx), len(eval_idx)
    centroid_mat = torch.stack([F.normalize(centroids[cls].float(), dim=0) for cls in used], dim=0)
    eval_h = F.normalize(hidden[eval_keep].float(), dim=1)
    pred = centroid_mat.matmul(eval_h.T).argmax(dim=0)
    pred_labels = [used[int(i)] for i in pred]
    acc = sum(1 for p, idx in zip(pred_labels, eval_keep) if p == labels[idx]) / len(eval_keep)
    return acc, centroids, len(train_idx), len(eval_idx)


def rotated_labels(labels: Sequence[int]) -> list[int]:
    classes = sorted(set(labels))
    if len(classes) < 2:
        return list(labels)
    nxt = {cls: classes[(i + 1) % len(classes)] for i, cls in enumerate(classes)}
    return [nxt[x] for x in labels]


def run_probe_selectivity(
    ctx: bench.RunContext,
    cfg: Lab29Config,
    model: Any,
    snapshots: Mapping[int, Mapping[str, Any]],
    stoi: Mapping[str, int],
    device: Any,
) -> tuple[list[dict[str, Any]], dict[int, dict[int, dict[int, Any]]]]:
    examples = generate_probe_examples(stoi, n_train=cfg.probe_train, n_eval=cfg.probe_eval, seed=cfg.seed)
    rows: list[dict[str, Any]] = []
    centroid_state: dict[int, dict[int, dict[int, Any]]] = {}
    for step in sorted(snapshots):
        model.load_state_dict(snapshots[step])
        model.eval()
        hidden_by_depth = collect_hidden_by_depth(model, examples, device)
        centroid_state[step] = {}
        for depth, (hidden, labels, splits) in hidden_by_depth.items():
            train_labels = [labels[i] for i, split in enumerate(splits) if split == "probe_train"]
            acc, centroids, n_train, n_eval = centroid_accuracy(hidden, labels, splits)
            ctrl_acc, _ctrl_centroids, _, _ = centroid_accuracy(hidden, labels, splits, train_labels_override=rotated_labels(train_labels))
            chance = 1.0 / max(1, len(set(train_labels)))
            centroid_state[step][depth] = centroids
            rows.append({
                "checkpoint_step": step,
                "depth": depth,
                "probe_acc": rounded(acc),
                "shuffled_label_acc": rounded(ctrl_acc),
                "chance_acc": rounded(chance),
                "selectivity": rounded(acc - ctrl_acc if math.isfinite(acc) and math.isfinite(ctrl_acc) else float("nan")),
                "selectivity_vs_chance": rounded(acc - chance if math.isfinite(acc) else float("nan")),
                "n_probe_train": n_train,
                "n_probe_eval": n_eval,
                "n_classes": len(centroids),
            })
    path = ctx.path("tables", "checkpoint_probe_selectivity.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Centroid-probe decodability and rotated-label controls by checkpoint and stream depth.")
    return rows, centroid_state


def cosine(a: Any, b: Any) -> float:
    import torch.nn.functional as F

    return float(F.cosine_similarity(a.float(), b.float(), dim=0).detach().cpu())


def write_feature_lineage(ctx: bench.RunContext, centroid_state: Mapping[int, Mapping[int, Mapping[int, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    final_step = max(centroid_state)
    final_depths = centroid_state[final_step]
    for step, by_depth in sorted(centroid_state.items()):
        for depth, centroids in sorted(by_depth.items()):
            best_final_depth = None
            best_cos = -float("inf")
            same_depth_cos = float("nan")
            for final_depth, final_centroids in final_depths.items():
                shared = sorted(set(centroids) & set(final_centroids))
                if not shared:
                    continue
                mean_cos = safe_mean([cosine(centroids[token], final_centroids[token]) for token in shared])
                if final_depth == depth:
                    same_depth_cos = mean_cos
                if mean_cos > best_cos:
                    best_cos = mean_cos
                    best_final_depth = final_depth
            rows.append({
                "checkpoint_step": step,
                "depth": depth,
                "same_depth_cosine_to_final": rounded(same_depth_cos),
                "best_matching_final_depth": "" if best_final_depth is None else best_final_depth,
                "best_final_depth_cosine": rounded(best_cos),
                "centroid_classes": len(centroids),
                "lineage_caveat": "cosine stability is not feature identity",
            })
    path = ctx.path("tables", "feature_lineage.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Centroid lineage similarity across checkpoints and depths.")
    return rows


def random_like(vector: Any, key: str) -> Any:
    import torch

    gen = torch.Generator(device="cpu").manual_seed(stable_int(key) % (2**31 - 1))
    rand = torch.randn(vector.shape, generator=gen, dtype=vector.dtype)
    return rand / rand.float().norm().clamp_min(1e-8) * vector.float().norm().clamp_min(1e-8)


def select_intervention_depth(probe_rows: Sequence[Mapping[str, Any]], final_step: int) -> int:
    final_probe_rows = [r for r in probe_rows if int(r["checkpoint_step"]) == final_step and int(r["depth"]) > 0]
    if not final_probe_rows:
        final_probe_rows = [r for r in probe_rows if int(r["checkpoint_step"]) == final_step]
    return int(max(final_probe_rows, key=lambda r: row_float(r, "selectivity", -999.0))["depth"])


def centroid_reference_norm(centroids: Mapping[int, Any]) -> float:
    vals = [float(v.float().norm()) for v in centroids.values()]
    return safe_median(vals, default=1.0)


def unit_scaled(direction: Any, scale: float) -> Any:
    return direction.float() / direction.float().norm().clamp_min(1e-8) * float(scale)


def run_intervention_transfer(
    ctx: bench.RunContext,
    model: Any,
    snapshots: Mapping[int, Mapping[str, Any]],
    tasks: Sequence[DynamicsTask],
    centroid_state: Mapping[int, Mapping[int, Mapping[int, Any]]],
    probe_rows: Sequence[Mapping[str, Any]],
    device: Any,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    final_step = max(snapshots)
    best_depth = select_intervention_depth(probe_rows, final_step)
    final_centroids = centroid_state[final_step][best_depth]
    ref_norm = centroid_reference_norm(final_centroids)
    scale = INTERVENTION_NORM_FRACTION * ref_norm
    rows: list[dict[str, Any]] = []
    for step in sorted(snapshots):
        model.load_state_dict(snapshots[step])
        model.eval()
        for task in tasks:
            if task.task_family != "induction_copy":
                continue
            if task.target_id not in final_centroids or task.distractor_id not in final_centroids:
                continue
            direction = unit_scaled(final_centroids[task.target_id] - final_centroids[task.distractor_id], scale)
            rand = random_like(direction, f"{step}|{task.item_id}|intervention")
            base_logits, _, _ = run_model_on_prompt(model, task.prompt_ids, device, return_cache=False)
            edited_logits, _, _ = run_model_on_prompt(model, task.prompt_ids, device, return_cache=False, addition=(best_depth, direction, 1.0))
            random_logits, _, _ = run_model_on_prompt(model, task.prompt_ids, device, return_cache=False, addition=(best_depth, rand, 1.0))
            base = logit_margin(base_logits, task.target_id, task.distractor_id)
            edited = logit_margin(edited_logits, task.target_id, task.distractor_id)
            random_margin = logit_margin(random_logits, task.target_id, task.distractor_id)
            rows.append({
                "checkpoint_step": step,
                "item_id": task.item_id,
                "split": task.split,
                "depth": best_depth,
                "intervention_norm_fraction": INTERVENTION_NORM_FRACTION,
                "absolute_scale": rounded(scale),
                "base_margin": rounded(base),
                "edited_margin": rounded(edited),
                "random_direction_margin": rounded(random_margin),
                "intervention_gain": rounded(edited - base),
                "random_gain": rounded(random_margin - base),
                "control_gap": rounded((edited - base) - (random_margin - base)),
                "base_correct": base > 0,
                "edited_correct": edited > 0,
                "random_correct": random_margin > 0,
            })
    path = ctx.path("tables", "intervention_transfer.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Final-checkpoint centroid-direction activation additions transferred to earlier checkpoints.")
    meta = {"intervention_depth": best_depth, "reference_norm": ref_norm, "absolute_scale": scale, "scale_semantics": "unit centroid direction multiplied by intervention_norm_fraction * median final-centroid norm"}
    return rows, best_depth, meta


# ---------------------------------------------------------------------------
# Summaries and artifacts
# ---------------------------------------------------------------------------


def split_accuracy(rows: Sequence[Mapping[str, Any]], split: str) -> float:
    return bool_mean([r for r in rows if r.get("split") == split], "correct")


def classify_phase(row: Mapping[str, Any], prev_best_depth: int | None) -> str:
    behavior = row_float(row, "induction_accuracy") >= BEHAVIOR_ACC_BAR
    heldout = row_float(row, "heldout_or_test_accuracy", 0.0) >= BEHAVIOR_ACC_BAR
    probe = row_float(row, "best_probe_acc") >= PROBE_ACC_BAR and row_float(row, "best_probe_selectivity") >= PROBE_SELECTIVITY_BAR
    motif = row_float(row, "mean_motif_control_gap") >= MOTIF_GAP_BAR
    depth = int(row.get("best_probe_depth", -1))
    if not behavior and not probe and not motif:
        return "absent_or_random"
    if probe and not behavior:
        return "decodable_before_behavioral"
    if behavior and not probe:
        return "behavioral_before_decodable"
    if behavior and probe and prev_best_depth is not None and depth != prev_best_depth:
        return "migration"
    if behavior and heldout and probe and motif:
        return "circuit_present_under_proxy"
    if behavior and probe and not motif:
        return "behavioral_decodable_no_mean_attention_motif"
    return "sharpening_or_redistributed"


def summarize_circuits(
    ctx: bench.RunContext,
    behavior_rows: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    intervention_rows: Sequence[Mapping[str, Any]],
    tasks: Sequence[DynamicsTask],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prev_best_depth: int | None = None
    induction_count = sum(1 for t in tasks if t.task_family == "induction_copy")
    checkpoints = sorted({int(r["checkpoint_step"]) for r in behavior_rows})
    for step in checkpoints:
        b_rows = [r for r in behavior_rows if int(r["checkpoint_step"]) == step and r["task_family"] == "induction_copy"]
        control_rows = [r for r in behavior_rows if int(r["checkpoint_step"]) == step and r["task_family"] != "induction_copy"]
        p_rows = [r for r in probe_rows if int(r["checkpoint_step"]) == step]
        i_rows = [r for r in intervention_rows if int(r["checkpoint_step"]) == step]
        best_probe = max(p_rows, key=lambda r: row_float(r, "selectivity", -999.0))
        motif_rows = [r for r in b_rows if r.get("mean_induction_source_attention") != ""]
        best_motif = max(motif_rows, key=lambda r: row_float(r, "best_induction_source_attention", -999.0)) if motif_rows else {}
        random_attention = safe_mean([r.get("random_attention_baseline") for r in motif_rows])
        mean_motif = safe_mean([r.get("mean_induction_source_attention") for r in motif_rows])
        motif_gap = mean_motif - random_attention if math.isfinite(mean_motif) and math.isfinite(random_attention) else float("nan")
        behavior_acc = bool_mean(b_rows, "correct", default=0.0)
        train_acc = split_accuracy(b_rows, "train")
        heldout_acc = split_accuracy(b_rows, "heldout")
        test_acc = split_accuracy(b_rows, "test")
        heldout_or_test = bool_mean([r for r in b_rows if r.get("split") in {"heldout", "test"}], "correct", default=0.0)
        control_acc = bool_mean(control_rows, "correct", default=0.0)
        intervention_gain = safe_mean([r.get("intervention_gain") for r in i_rows])
        random_gain = safe_mean([r.get("random_gain") for r in i_rows])
        intervention_gap = intervention_gain - random_gain if math.isfinite(intervention_gain) and math.isfinite(random_gain) else float("nan")
        draft = {
            "checkpoint_step": step,
            "induction_accuracy": rounded(behavior_acc),
            "train_accuracy_eval_rows": rounded(train_acc),
            "heldout_accuracy": rounded(heldout_acc),
            "test_accuracy": rounded(test_acc),
            "heldout_or_test_accuracy": rounded(heldout_or_test),
            "control_accuracy": rounded(control_acc),
            "control_leakage_flag": False,
            "mean_induction_margin": rounded(safe_mean([r.get("target_minus_distractor") for r in b_rows])),
            "mean_control_margin": rounded(safe_mean([r.get("target_minus_distractor") for r in control_rows])),
            "best_probe_depth": int(best_probe["depth"]),
            "best_probe_acc": best_probe["probe_acc"],
            "best_probe_selectivity": best_probe["selectivity"],
            "best_probe_shuffled_label_acc": best_probe["shuffled_label_acc"],
            "best_probe_chance_acc": best_probe["chance_acc"],
            "best_induction_source_attention": rounded(best_motif.get("best_induction_source_attention", float("nan"))),
            "mean_induction_source_attention": rounded(mean_motif),
            "best_prev_match_attention": rounded(best_motif.get("best_induction_source_attention", float("nan"))),
            "mean_prev_match_attention": rounded(mean_motif),
            "mean_random_attention_baseline": rounded(random_attention),
            "mean_motif_control_gap": rounded(motif_gap),
            "best_motif_layer": best_motif.get("best_motif_layer", ""),
            "best_motif_head": best_motif.get("best_motif_head", ""),
            "mean_intervention_gain": rounded(intervention_gain),
            "mean_random_gain": rounded(random_gain),
            "intervention_control_gap": rounded(intervention_gap),
            "n_induction_tasks": induction_count,
            "n_control_tasks": len({r.get("item_id") for r in control_rows}),
        }
        draft["phase"] = classify_phase(draft, prev_best_depth)
        rows.append(draft)
        prev_best_depth = int(best_probe["depth"])

    if rows:
        baseline_control_acc = row_float(rows[0], "control_accuracy", 0.0)
        baseline_control_margin = row_float(rows[0], "mean_control_margin", 0.0)
        first_step = int(rows[0]["checkpoint_step"])
        for row in rows:
            acc_delta = row_float(row, "control_accuracy", 0.0) - baseline_control_acc
            margin_delta = row_float(row, "mean_control_margin", 0.0) - baseline_control_margin
            row["control_accuracy_delta_from_step0"] = rounded(acc_delta)
            row["mean_control_margin_delta_from_step0"] = rounded(margin_delta)
            row["control_leakage_flag"] = bool(
                int(row["checkpoint_step"]) > first_step
                and acc_delta >= 0.25
                and margin_delta >= 0.20
                and row_float(row, "control_accuracy", 0.0) >= CONTROL_LEAKAGE_ACC_BAR
            )

    event_specs = [
        ("behavior_emergence", lambda r: row_float(r, "induction_accuracy") >= BEHAVIOR_ACC_BAR and row_float(r, "heldout_or_test_accuracy", 0.0) >= BEHAVIOR_ACC_BAR, f"induction and heldout/test accuracy >= {BEHAVIOR_ACC_BAR}"),
        ("decodability_emergence", lambda r: row_float(r, "best_probe_acc") >= PROBE_ACC_BAR and row_float(r, "best_probe_selectivity") >= PROBE_SELECTIVITY_BAR, f"probe acc >= {PROBE_ACC_BAR} and selectivity >= {PROBE_SELECTIVITY_BAR}"),
        ("mean_attention_motif_emergence", lambda r: row_float(r, "mean_motif_control_gap") >= MOTIF_GAP_BAR, f"mean previous-match-successor attention gap >= {MOTIF_GAP_BAR}"),
        ("intervention_transfer_emergence", lambda r: row_float(r, "intervention_control_gap") >= INTERVENTION_GAP_BAR and row_float(r, "induction_accuracy") >= BEHAVIOR_ACC_BAR and row_float(r, "heldout_or_test_accuracy", 0.0) >= BEHAVIOR_ACC_BAR and row_float(r, "best_probe_acc") >= PROBE_ACC_BAR and row_float(r, "best_probe_selectivity") >= PROBE_SELECTIVITY_BAR, f"intervention gain beats random by >= {INTERVENTION_GAP_BAR} after behavior, heldout/test, and probe gates"),
        ("control_task_leakage", lambda r: bool(r.get("control_leakage_flag")), f"untrained control accuracy improves by >=0.25 and margin by >=0.20 over checkpoint zero, with accuracy >= {CONTROL_LEAKAGE_ACC_BAR}; this is a failure event"),
    ]
    events: list[dict[str, Any]] = []
    for event, pred, threshold in event_specs:
        hits = [r for r in rows if pred(r)]
        first = hits[0] if hits else None
        failure = event == "control_task_leakage"
        events.append({
            "event": event,
            "first_checkpoint_step": "" if first is None else first["checkpoint_step"],
            "threshold": threshold,
            "observed_values_json": "" if first is None else json.dumps({k: first[k] for k in ("induction_accuracy", "heldout_or_test_accuracy", "best_probe_acc", "best_probe_selectivity", "mean_motif_control_gap", "intervention_control_gap", "control_accuracy")}),
            "claim_status": ("failure_observed" if failure and first is not None else "not_observed" if first is None else "observed_in_controlled_sequence"),
            "claim_caveat": "failure event; narrows or blocks training-dynamics claim" if failure else "threshold crossing, not exact birth step",
        })

    counterexamples: list[dict[str, Any]] = []
    for row in rows:
        step = row["checkpoint_step"]
        if row_float(row, "best_probe_shuffled_label_acc") >= row_float(row, "best_probe_acc") - CLOSE_CONTROL_TOL:
            counterexamples.append({"checkpoint_step": step, "kind": "probe_control_close", "metric": "shuffled_label_acc", "observed": row["best_probe_shuffled_label_acc"], "claim_pressure": "decodability", "lesson": "The rotated-label control is close to the real probe; do not claim readable target identity at this checkpoint."})
        if bool(row.get("control_leakage_flag")):
            counterexamples.append({"checkpoint_step": step, "kind": "untrained_control_leakage", "metric": "control_accuracy_delta_from_step0", "observed": row.get("control_accuracy_delta_from_step0", ""), "claim_pressure": "specificity", "lesson": "An untrained control family improved relative to checkpoint zero enough to threaten a task-specific training story."})
        if row_float(row, "intervention_control_gap") >= INTERVENTION_GAP_BAR and (row_float(row, "induction_accuracy") < BEHAVIOR_ACC_BAR or row_float(row, "heldout_or_test_accuracy", 0.0) < BEHAVIOR_ACC_BAR or row_float(row, "best_probe_acc") < PROBE_ACC_BAR or row_float(row, "best_probe_selectivity") < PROBE_SELECTIVITY_BAR):
            counterexamples.append({"checkpoint_step": step, "kind": "intervention_too_early", "metric": "intervention_control_gap", "observed": row["intervention_control_gap"], "claim_pressure": "causal_birth", "lesson": "A final-checkpoint direction worked before behavior/probe prerequisites; treat as coordinate alignment, not circuit birth."})
    if rows:
        final = rows[-1]
        if row_float(final, "mean_motif_control_gap") < MOTIF_GAP_BAR and row_float(final, "induction_accuracy") >= BEHAVIOR_ACC_BAR:
            counterexamples.append({"checkpoint_step": final["checkpoint_step"], "kind": "behavior_without_mean_attention_motif", "metric": "mean_motif_control_gap", "observed": final["mean_motif_control_gap"], "claim_pressure": "attention_motif", "lesson": "Behavior is present but the mean previous-match-successor attention motif did not clear the bar."})

    path = ctx.path("tables", "checkpoint_circuit_summary.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Checkpoint-level phase classification and circuit summary.")
    event_path = ctx.path("tables", "mechanism_birth_events.csv")
    bench.write_csv_with_context(ctx, event_path, events)
    ctx.register_artifact(event_path, "table", "Thresholded behavior/decoding/motif/intervention event ledger.")
    cex_path = ctx.path("tables", "training_dynamics_counterexamples.csv")
    bench.write_csv_with_context(ctx, cex_path, counterexamples)
    ctx.register_artifact(cex_path, "table", "Rows that block or narrow circuit-birth language.")

    metrics = {
        "n_checkpoints": len(rows),
        "n_induction_tasks": induction_count,
        "n_counterexamples": len(counterexamples),
        "final_induction_accuracy": rows[-1]["induction_accuracy"] if rows else "",
        "final_heldout_or_test_accuracy": rows[-1]["heldout_or_test_accuracy"] if rows else "",
        "final_probe_acc": rows[-1]["best_probe_acc"] if rows else "",
        "final_probe_selectivity": rows[-1]["best_probe_selectivity"] if rows else "",
        "final_phase": rows[-1]["phase"] if rows else "",
        "behavior_emergence_step": next((r["first_checkpoint_step"] for r in events if r["event"] == "behavior_emergence"), ""),
        "decodability_emergence_step": next((r["first_checkpoint_step"] for r in events if r["event"] == "decodability_emergence"), ""),
        "motif_emergence_step": next((r["first_checkpoint_step"] for r in events if r["event"] == "mean_attention_motif_emergence"), ""),
        "intervention_transfer_step": next((r["first_checkpoint_step"] for r in events if r["event"] == "intervention_transfer_emergence"), ""),
        "control_leakage_step": next((r["first_checkpoint_step"] for r in events if r["event"] == "control_task_leakage"), ""),
        "thresholds": {
            "behavior_acc_bar": BEHAVIOR_ACC_BAR,
            "probe_acc_bar": PROBE_ACC_BAR,
            "probe_selectivity_bar": PROBE_SELECTIVITY_BAR,
            "motif_gap_bar": MOTIF_GAP_BAR,
            "intervention_gap_bar": INTERVENTION_GAP_BAR,
            "control_leakage_acc_bar": CONTROL_LEAKAGE_ACC_BAR,
        },
    }
    return rows, events, counterexamples, metrics


def write_evidence_matrix(ctx: bench.RunContext, circuit_rows: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    final = circuit_rows[-1] if circuit_rows else {}
    event_by_name = {str(e["event"]): e for e in events}
    rows = [
        {"evidence_component": "behavior", "rung": "OBS", "final_value": final.get("induction_accuracy", ""), "control_value": final.get("control_accuracy", ""), "first_checkpoint": event_by_name.get("behavior_emergence", {}).get("first_checkpoint_step", ""), "gate_passed": bool(event_by_name.get("behavior_emergence", {}).get("first_checkpoint_step", "") != ""), "claim_allowed": "behavior threshold ordering only"},
        {"evidence_component": "decodability", "rung": "DECODE", "final_value": final.get("best_probe_acc", ""), "control_value": final.get("best_probe_shuffled_label_acc", ""), "first_checkpoint": event_by_name.get("decodability_emergence", {}).get("first_checkpoint_step", ""), "gate_passed": bool(event_by_name.get("decodability_emergence", {}).get("first_checkpoint_step", "") != ""), "claim_allowed": "target identity is linearly readable at selected depths"},
        {"evidence_component": "attention_motif", "rung": "ATTR", "final_value": final.get("mean_motif_control_gap", ""), "control_value": 0.0, "first_checkpoint": event_by_name.get("mean_attention_motif_emergence", {}).get("first_checkpoint_step", ""), "gate_passed": bool(event_by_name.get("mean_attention_motif_emergence", {}).get("first_checkpoint_step", "") != ""), "claim_allowed": "previous-match-successor attention motif is visible on average"},
        {"evidence_component": "intervention_transfer", "rung": "CAUSAL", "final_value": final.get("intervention_control_gap", ""), "control_value": final.get("mean_random_gain", ""), "first_checkpoint": event_by_name.get("intervention_transfer_emergence", {}).get("first_checkpoint_step", ""), "gate_passed": bool(event_by_name.get("intervention_transfer_emergence", {}).get("first_checkpoint_step", "") != ""), "claim_allowed": "final centroid direction has a scoped causal handle after prerequisites"},
        {"evidence_component": "control_leakage", "rung": "AUDIT", "final_value": final.get("control_accuracy_delta_from_step0", ""), "control_value": "+0.25 accuracy and +0.20 margin drift", "first_checkpoint": event_by_name.get("control_task_leakage", {}).get("first_checkpoint_step", ""), "gate_passed": not bool(event_by_name.get("control_task_leakage", {}).get("first_checkpoint_step", "")), "claim_allowed": "specificity audit passes only if untrained controls do not drift"},
    ]
    for rel, desc in [
        (("tables", "training_dynamics_evidence_matrix.csv"), "Evidence matrix for behavior, decodability, motif, intervention, and control leakage."),
        (("tables", "evidence_matrix.csv"), "Standard alias for the Lab 29 evidence matrix."),
    ]:
        path = ctx.path(*rel)
        bench.write_csv_with_context(ctx, path, rows)
        ctx.register_artifact(path, "table", desc)
    return rows


def write_task_manifest(ctx: bench.RunContext, tasks: Sequence[DynamicsTask], vocab: Sequence[str]) -> None:
    rows = [{
        "item_id": t.item_id,
        "task_family": t.task_family,
        "prompt": t.prompt,
        "prompt_ids": " ".join(str(i) for i in t.prompt_ids),
        "prompt_tokens": " ".join(vocab[i] for i in t.prompt_ids),
        "target": t.target,
        "target_id": t.target_id,
        "distractor": t.distractor,
        "distractor_id": t.distractor_id,
        "split": t.split,
        "expected_mechanism": t.expected_mechanism,
        "notes": t.notes,
    } for t in tasks]
    path = ctx.path("tables", "task_manifest.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Selected Lab 29 task rows and tiny-vocabulary encodings.")


def save_state(
    ctx: bench.RunContext,
    cfg: Lab29Config,
    vocab: Sequence[str],
    snapshots: Mapping[int, Mapping[str, Any]],
    centroid_state: Mapping[int, Mapping[int, Mapping[int, Any]]],
    intervention_depth: int,
    intervention_meta: Mapping[str, Any],
) -> None:
    import torch

    payload = {
        "lab": LAB_ID,
        "vocab": list(vocab),
        "checkpoint_steps": sorted(snapshots),
        "tiny_model_config": {
            "d_model": TINY_D_MODEL,
            "n_layers": TINY_LAYERS,
            "n_heads": TINY_HEADS,
            "max_len": TINY_MAX_LEN,
            "task": "synthetic induction-copy next-token training",
        },
        "training_config": dataclasses.asdict(cfg),
        "state_dicts": dict(snapshots),
        "centroids": dict(centroid_state),
        "intervention_depth": intervention_depth,
        "intervention_meta": dict(intervention_meta),
        "note": "Controlled tiny-model checkpoints; do not compare these steps to pretrained-model training tokens.",
    }
    path = ctx.path("state", "checkpoint_directions.pt")
    torch.save(payload, path)
    ctx.register_artifact(path, "state", "Tiny checkpoint sequence, centroid directions, and intervention depth.")
    meta_path = ctx.path("state", "checkpoint_directions_metadata.json")
    bench.write_json(meta_path, {
        "lab": LAB_ID,
        "checkpoint_steps": sorted(snapshots),
        "vocab": list(vocab),
        "tiny_model_config": payload["tiny_model_config"],
        "training_config": dataclasses.asdict(cfg),
        "intervention_depth": intervention_depth,
        "intervention_meta": dict(intervention_meta),
        "stream_depth_convention": "depth 0 = token+position embedding stream; depth k>0 = residual stream after k tiny transformer blocks, before final_ln readout",
    })
    ctx.register_artifact(meta_path, "state", "Human-readable metadata for checkpoint_directions.pt.")


def run_tiny_self_checks(ctx: bench.RunContext, model: Any, snapshots: Mapping[int, Mapping[str, Any]], tasks: Sequence[DynamicsTask], device: Any) -> dict[str, Any]:
    import torch

    task = next((t for t in tasks if t.task_family == "induction_copy"), tasks[0])
    first_step = min(snapshots)
    final_step = max(snapshots)
    model.load_state_dict(snapshots[first_step])
    logits_a, _, _ = run_model_on_prompt(model, task.prompt_ids, device, return_cache=False)
    model.load_state_dict(snapshots[first_step])
    logits_b, _, _ = run_model_on_prompt(model, task.prompt_ids, device, return_cache=False)
    reload_max_diff = float((logits_a - logits_b).abs().max())
    zero_vec = torch.zeros(TINY_D_MODEL)
    logits_noop, _, _ = run_model_on_prompt(model, task.prompt_ids, device, return_cache=False, addition=(1, zero_vec, 0.0))
    noop_max_diff = float((logits_b - logits_noop).abs().max())
    model.load_state_dict(snapshots[final_step])
    logits_final_1, streams, attentions = run_model_on_prompt(model, task.prompt_ids, device, return_cache=True)
    logits_final_2, _, _ = run_model_on_prompt(model, task.prompt_ids, device, return_cache=True)
    determinism_max_diff = float((logits_final_1 - logits_final_2).abs().max())
    attention_shapes_ok = len(attentions) == TINY_LAYERS and all(a is not None and a.shape[1] == TINY_HEADS for a in attentions)
    stream_shapes_ok = len(streams) == TINY_LAYERS + 1 and all(s.shape[-1] == TINY_D_MODEL for s in streams)
    ok = reload_max_diff <= 1e-7 and noop_max_diff <= 1e-7 and determinism_max_diff <= 1e-7 and attention_shapes_ok and stream_shapes_ok
    result = {
        "checkpoint_reload_ok": reload_max_diff <= 1e-7,
        "checkpoint_reload_max_diff": reload_max_diff,
        "zero_addition_noop_ok": noop_max_diff <= 1e-7,
        "zero_addition_max_diff": noop_max_diff,
        "forward_determinism_ok": determinism_max_diff <= 1e-7,
        "forward_determinism_max_diff": determinism_max_diff,
        "attention_shapes_ok": attention_shapes_ok,
        "stream_shapes_ok": stream_shapes_ok,
        "n_checkpoints": len(snapshots),
        "ok": ok,
        "note": "Tiny-model self-checks are lab-local because Lab 29 trains its own model; the HF bench model is only the outer runner.",
    }
    path = ctx.path("diagnostics", "self_check_status.json")
    bench.write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "Lab-local self-checks for tiny checkpoint reload, no-op intervention, determinism, and tensor shapes.")
    return result


def write_status_files(ctx: bench.RunContext, data_info: Mapping[str, Any], self_checks: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    science_ready = bool(data_info.get("science_ready")) and bool(self_checks.get("ok")) and metrics.get("n_checkpoints", 0) > 1
    safety = {
        "lab": "lab29",
        "unsafe_prompt_sampling": False,
        "harmful_completion_generation": False,
        "external_checkpoint_claim": False,
        "trains_model_during_run": True,
        "training_data": "synthetic induction-copy only",
        "science_ready": science_ready,
        "note": "The lab trains a tiny transformer from scratch on benign synthetic sequences and evaluates forward-pass diagnostics.",
    }
    path = ctx.path("diagnostics", "safety_status.json")
    bench.write_json(path, safety)
    ctx.register_artifact(path, "diagnostic", "Safety and scope status for Lab 29.")


def write_plot_guide(ctx: bench.RunContext) -> None:
    rows = [
        {"plot": "training_dynamics_dashboard.png", "first_question": "Which measurement crossed first?", "claim_boundary": "Threshold order, not exact birth."},
        {"plot": "behavior_vs_decodability_timeline.png", "first_question": "Did behavior and decodability emerge together?", "claim_boundary": "Probe readability is not use."},
        {"plot": "circuit_birth_atlas.png", "first_question": "Which gates are present at each checkpoint?", "claim_boundary": "Column scales are different measurement families."},
        {"plot": "depth_migration_map.png", "first_question": "Did the best readable depth move?", "claim_boundary": "Best-depth summaries are not feature identities."},
        {"plot": "checkpoint_feature_lineage.png", "first_question": "Do final-checkpoint centroids align with earlier centroids?", "claim_boundary": "Cosine stability is not sameness."},
        {"plot": "intervention_transfer_over_time.png", "first_question": "When does the final direction beat random?", "claim_boundary": "Tiny checkpoints share coordinates by construction."},
        {"plot": "random_model_control_panel.png", "first_question": "Did untrained controls stay flat relative to checkpoint zero?", "claim_boundary": "Control drift blocks specificity language."},
        {"plot": "tiny_training_curve.png", "first_question": "Did training actually optimize the synthetic task?", "claim_boundary": "Batch training accuracy is not held-out behavior."},
    ]
    path = ctx.path("tables", "plot_reading_guide.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Map from Lab 29 plots to the question and claim boundary each protects.")


def write_cards(
    ctx: bench.RunContext,
    cfg: Lab29Config,
    data_info: Mapping[str, Any],
    metrics: Mapping[str, Any],
    circuit_rows: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
    self_checks: Mapping[str, Any],
) -> None:
    event_lines = ["| event | first checkpoint | status | caveat |", "|---|---:|---|---|"]
    for row in events:
        event_lines.append(f"| {row['event']} | {row['first_checkpoint_step']} | {row['claim_status']} | {row['claim_caveat']} |")
    final = circuit_rows[-1] if circuit_rows else {}
    method = [
        "# Lab 29 method card", "",
        "This lab uses a tiny transformer trained inside the run. It is a controlled time-lapse, not a Pythia or OLMo training-history result.", "",
        f"- training task: synthetic induction-copy sequences over {len(INDUCTION_TOKENS)} color tokens",
        f"- architecture: {TINY_LAYERS} layers, {TINY_HEADS} heads, d_model={TINY_D_MODEL}",
        f"- train steps: {cfg.train_steps}; checkpoints: {cfg.checkpoint_steps}",
        f"- data source: `{data_info.get('data_source')}`; science_ready: `{str(data_info.get('science_ready')).lower()}`",
        "- controls: untrained task-family drift relative to checkpoint zero, rotated-label centroid probe, random-direction intervention, checkpoint-zero baseline",
        "- evidence rung: `OBS + DECODE + ATTR`, plus scoped toy `CAUSAL` for intervention transfer",
        "- forbidden claim: the model first learned a concept at exactly this step", "",
        "## Birth-event ledger", "", *event_lines, "",
        "## Final checkpoint", "",
        "| step | induction acc | heldout/test acc | probe acc | selectivity | motif gap | intervention gap | phase |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    if final:
        method.append(f"| {final['checkpoint_step']} | {final['induction_accuracy']} | {final['heldout_or_test_accuracy']} | {final['best_probe_acc']} | {final['best_probe_selectivity']} | {final['mean_motif_control_gap']} | {final['intervention_control_gap']} | {final['phase']} |")
    method += ["", "Safe sentence: `In this controlled checkpoint sequence, these measurement thresholds crossed in this order under the listed controls.`", "", "Unsafe sentence: `The model first learned induction at exactly checkpoint K.`", ""]
    path = ctx.path("method_card.md")
    bench.write_text(path, "\n".join(method))
    ctx.register_artifact(path, "summary", "Lab 29 method card and event summary.")

    audit_result = "failed" if metrics.get("control_leakage_step") else ("mixed" if counterexamples else "passed")
    audit = [
        "# Lab 29 operationalization audit", "",
        "```yaml",
        "headline_claim: \"a circuit is born during training\"",
        "cheap_explanation: \"a threshold crossing, probe artifact, control leakage, or coordinate-aligned intervention is being mistaken for birth\"",
        "killer_control: \"rotated-label probes, untrained tasks, random-direction interventions, checkpoint-zero baselines, and counterexample rows\"",
        f"result: \"{audit_result}\"",
        "claim_allowed: \"threshold ordering handle\"",
        "```", "",
        "## What this run can say", "",
        "It can order behavior, decodability, attention-motif, and intervention-transfer threshold crossings for one controlled tiny checkpoint sequence.", "",
        "## What it cannot say", "",
        "It cannot identify an exact learning instant, prove feature identity across checkpoints, or transfer tiny-model results to pretrained LLMs without rerunning the audit on those checkpoints.", "",
        "## Counterexamples", "",
    ]
    if counterexamples:
        audit += [f"- step `{r['checkpoint_step']}` `{r['kind']}`: {r['lesson']}" for r in counterexamples]
    else:
        audit.append("- No automatic counterexample crossed the current thresholds. This is not replication; inspect the raw controls before broadening the claim.")
    audit += ["", "## Phase trajectory", ""]
    for row in circuit_rows:
        audit.append(f"- step `{row['checkpoint_step']}`: `{row['phase']}` (behavior={row['induction_accuracy']}, probe={row['best_probe_acc']}, motif_gap={row['mean_motif_control_gap']}, intervention_gap={row['intervention_control_gap']}).")
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(audit) + "\n")
    ctx.register_artifact(path, "summary", "Controls and non-claims for Lab 29 training dynamics.")

    science_ready = bool(data_info.get("science_ready")) and bool(self_checks.get("ok")) and not bool(metrics.get("control_leakage_step"))
    smallest_claim = "No broad circuit-birth claim survived the automatic audit." if not science_ready else "A controlled threshold-ordering claim is available; exact birth language is still forbidden."
    if data_info.get("fallback_warning"):
        smallest_claim = "Smoke-only fallback data was used; do not ledger a science claim."
    main_counter = counterexamples[0]["lesson"] if counterexamples else "No automatic counterexample crossed thresholds; controls still need human inspection."
    summary = [
        "# Lab 29 run summary: training dynamics and circuit birth", "",
        f"- data rows: {data_info['n_rows_selected']} selected from `{pathlib.Path(str(data_info['data_path'])).name}`",
        f"- data source: `{data_info.get('data_source')}`",
        f"- science_ready: `{str(science_ready).lower()}`",
        f"- tiny train steps: {cfg.train_steps}; checkpoints: {cfg.checkpoint_steps}",
        f"- final induction accuracy: `{metrics.get('final_induction_accuracy')}`",
        f"- final heldout/test accuracy: `{metrics.get('final_heldout_or_test_accuracy')}`",
        f"- final probe selectivity: `{metrics.get('final_probe_selectivity')}`",
        f"- final phase: `{metrics.get('final_phase')}`",
        f"- smallest surviving claim: {smallest_claim}",
        f"- main counterexample: {main_counter}", "",
        "## Birth-event ledger", "", *event_lines, "",
        "## Reading order", "",
        "1. `method_card.md` for the scope and event thresholds.",
        "2. `diagnostics/self_check_status.json` and `diagnostics/tokenization_gate.csv` before trusting any science plot.",
        "3. `tables/checkpoint_behavior.csv` for margins, logit lens, and attention motifs.",
        "4. `tables/checkpoint_probe_selectivity.csv` for decodability and rotated-label controls.",
        "5. `tables/checkpoint_circuit_summary.csv` and `tables/mechanism_birth_events.csv` for threshold order.",
        "6. `tables/training_dynamics_counterexamples.csv` and `operationalization_audit.md` before writing claims.", "",
        "## Caveat", "",
        "A threshold crossing is a measurement event. It is not the model's first internal use of a concept, and this tiny setup does not stand in for pretrained LLM training dynamics.", "",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(summary))
    ctx.register_artifact(path, "summary", "Run summary, reading order, and smallest surviving claim.")


def write_claims(ctx: bench.RunContext, events: Sequence[Mapping[str, Any]], circuit_rows: Sequence[Mapping[str, Any]], data_info: Mapping[str, Any]) -> None:
    run_name = ctx.run_dir.name
    final = circuit_rows[-1] if circuit_rows else {}
    claims = []
    for i, row in enumerate(events, start=1):
        tag = "AUDIT" if row["event"] == "control_task_leakage" else ("OBS,DECODE,CAUSAL" if row["event"] == "intervention_transfer_emergence" else "OBS,DECODE,ATTR")
        text = (
            f"In Lab 29's controlled tiny checkpoint sequence, `{row['event']}` first crossed the lab threshold at checkpoint `{row['first_checkpoint_step']}`. "
            f"Status: {row['claim_status']}. Final phase: {final.get('phase', '')}. "
            "This is a threshold-ordering claim, not an exact learning-instant claim."
        )
        if data_info.get("fallback_warning"):
            text = "Smoke-only fallback data was used; this draft is an audit reminder rather than a science claim. " + text
        claims.append({
            "id": f"{LAB_ID}-C{i}",
            "tag": tag,
            "text": text,
            "artifact": f"runs/{run_name}/tables/mechanism_birth_events.csv",
            "falsifier": "Rotated-label probes, untrained controls, or random-direction interventions cross the same threshold, or the ordering fails on an independent checkpoint sequence.",
        })
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def write_placeholder(ctx: bench.RunContext, name: str, title: str, message: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10, wrap=True)
    bench.save_figure(ctx, fig, name, title)


def write_plots(
    ctx: bench.RunContext,
    circuit_rows: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    intervention_rows: Sequence[Mapping[str, Any]],
    behavior_rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
) -> None:
    write_plot_guide(ctx)
    if ctx.args.no_plots:
        return
    if not circuit_rows:
        for name in ("training_dynamics_dashboard.png", "behavior_vs_decodability_timeline.png", "circuit_birth_atlas.png", "depth_migration_map.png", "checkpoint_feature_lineage.png", "intervention_transfer_over_time.png", "random_model_control_panel.png", "tiny_training_curve.png"):
            write_placeholder(ctx, name, name.replace("_", " ").replace(".png", ""), "No checkpoint summary rows were produced.")
        return
    import matplotlib.pyplot as plt
    import numpy as np

    steps = [int(r["checkpoint_step"]) for r in circuit_rows]
    behavior = [row_float(r, "induction_accuracy", 0.0) for r in circuit_rows]
    heldout = [row_float(r, "heldout_or_test_accuracy", 0.0) for r in circuit_rows]
    probe = [row_float(r, "best_probe_acc", 0.0) for r in circuit_rows]
    selectivity = [row_float(r, "best_probe_selectivity", 0.0) for r in circuit_rows]
    motif = [row_float(r, "mean_motif_control_gap", 0.0) for r in circuit_rows]
    intervention = [row_float(r, "intervention_control_gap", 0.0) for r in circuit_rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Lab 29 training dynamics dashboard", fontsize=14, fontweight="bold")
    axes[0, 0].plot(steps, behavior, marker="o", label="all induction")
    axes[0, 0].plot(steps, heldout, marker="o", label="heldout/test")
    axes[0, 0].plot(steps, probe, marker="o", label="probe acc")
    axes[0, 0].axhline(BEHAVIOR_ACC_BAR, linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("Behavior vs decodability")
    axes[0, 0].set_xlabel("training step")
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].plot(steps, selectivity, marker="o", label="probe selectivity")
    axes[0, 1].plot(steps, motif, marker="o", label="mean motif gap")
    axes[0, 1].axhline(PROBE_SELECTIVITY_BAR, linestyle="--", linewidth=0.8)
    axes[0, 1].axhline(MOTIF_GAP_BAR, linestyle=":", linewidth=0.8)
    axes[0, 1].set_title("Decode and motif gates")
    axes[0, 1].set_xlabel("training step")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(steps, intervention, marker="o")
    axes[1, 0].axhline(INTERVENTION_GAP_BAR, linestyle="--", linewidth=0.8)
    axes[1, 0].set_title("Intervention transfer gap")
    axes[1, 0].set_xlabel("training step")

    phase_ids = {phase: i for i, phase in enumerate(sorted({str(r["phase"]) for r in circuit_rows}))}
    axes[1, 1].step(steps, [phase_ids[str(r["phase"])] for r in circuit_rows], where="mid")
    axes[1, 1].set_yticks(list(phase_ids.values()), list(phase_ids.keys()), fontsize=7)
    axes[1, 1].set_title("Phase trajectory")
    axes[1, 1].set_xlabel("training step")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    bench.save_figure(ctx, fig, "training_dynamics_dashboard.png", "Lab 29 training dynamics dashboard.")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(steps, behavior, marker="o", label="induction behavior")
    ax.plot(steps, heldout, marker="o", label="heldout/test behavior")
    ax.plot(steps, probe, marker="o", label="probe accuracy")
    ax.plot(steps, selectivity, marker="o", label="probe selectivity")
    ax.axhline(BEHAVIOR_ACC_BAR, linestyle="--", linewidth=0.8)
    ax.set_xlabel("training step")
    ax.set_ylabel("accuracy / selectivity")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Behavior vs decodability timeline")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "behavior_vs_decodability_timeline.png", "Behavior and decodability timeline.")

    grid = np.array([[row_float(r, "induction_accuracy", 0.0), row_float(r, "best_probe_selectivity", 0.0), row_float(r, "mean_motif_control_gap", 0.0), row_float(r, "intervention_control_gap", 0.0), row_float(r, "control_accuracy", 0.0)] for r in circuit_rows])
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    im = ax.imshow(grid, aspect="auto")
    ax.set_yticks(range(len(steps)), steps)
    ax.set_xticks(range(5), ["behavior", "selectivity", "motif gap", "intervention", "control acc"], rotation=20, ha="right")
    ax.set_title("Circuit birth atlas")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.035)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "circuit_birth_atlas.png", "Checkpoint metric atlas for Lab 29.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, [int(r["best_probe_depth"]) for r in circuit_rows], marker="o")
    ax.set_xlabel("training step")
    ax.set_ylabel("best probe depth")
    ax.set_title("Depth migration map")
    ax.set_yticks(range(TINY_LAYERS + 1))
    fig.tight_layout()
    bench.save_figure(ctx, fig, "depth_migration_map.png", "Best decodability depth over checkpoints.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for depth in sorted({int(r["depth"]) for r in feature_rows}):
        rows = [r for r in feature_rows if int(r["depth"]) == depth]
        xs = [int(r["checkpoint_step"]) for r in rows]
        ys = [row_float(r, "same_depth_cosine_to_final") for r in rows]
        ax.plot(xs, ys, marker="o", label=f"depth {depth}")
    ax.set_xlabel("training step")
    ax.set_ylabel("same-depth cosine to final")
    ax.set_title("Checkpoint feature lineage")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "checkpoint_feature_lineage.png", "Centroid lineage similarity to final checkpoint.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    gains_by_step = defaultdict(list)
    random_by_step = defaultdict(list)
    for row in intervention_rows:
        gains_by_step[int(row["checkpoint_step"])].append(row_float(row, "intervention_gain"))
        random_by_step[int(row["checkpoint_step"])].append(row_float(row, "random_gain"))
    ax.plot(steps, [safe_mean(gains_by_step[s]) for s in steps], marker="o", label="final direction")
    ax.plot(steps, [safe_mean(random_by_step[s]) for s in steps], marker="o", label="random direction")
    ax.set_xlabel("training step")
    ax.set_ylabel("margin gain")
    ax.set_title("Intervention transfer over time")
    ax.legend(frameon=False)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "intervention_transfer_over_time.png", "Final-direction intervention transfer over checkpoints.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    control = [r for r in behavior_rows if r["task_family"] != "induction_copy"]
    control_steps = sorted({int(r["checkpoint_step"]) for r in control})
    ax.plot(control_steps, [bool_mean([r for r in control if int(r["checkpoint_step"]) == s], "correct", default=0.0) for s in control_steps], marker="o", label="untrained controls")
    ax.plot(steps, behavior, marker="o", label="trained induction")
    ax.axhline(CONTROL_LEAKAGE_ACC_BAR, linestyle="--", linewidth=0.8, label="leakage bar")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("training step")
    ax.set_ylabel("accuracy")
    ax.set_title("Random/untrained control panel")
    ax.legend(frameon=False)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "random_model_control_panel.png", "Untrained-control behavior panel.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    train_numeric = [r for r in train_rows if r.get("loss") != ""]
    ax.plot([int(r["checkpoint_step"]) for r in train_numeric], [row_float(r, "loss") for r in train_numeric], marker="o", label="loss")
    ax2 = ax.twinx()
    ax2.plot([int(r["checkpoint_step"]) for r in train_numeric], [row_float(r, "train_accuracy") for r in train_numeric], marker="o", linestyle="--", label="batch acc")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax2.set_ylabel("batch accuracy")
    ax.set_title("Tiny training curve")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "tiny_training_curve.png", "Tiny transformer training curve.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    configure_torch_runtime()
    del bundle  # Lab 29 trains its own tiny model; the bench model is only the outer runner.
    cfg = build_config(ctx)
    tasks, data_info = load_tasks(ctx)
    stoi, vocab = build_vocab(tasks)
    tasks = tokenization_gate(ctx, tasks, stoi, vocab)
    write_task_manifest(ctx, tasks, vocab)

    manifest_path = ctx.path("diagnostics", "data_manifest.json")
    bench.write_json(manifest_path, {**data_info, "vocab_size": len(vocab), "vocab": list(vocab), "tiny_model": {"d_model": TINY_D_MODEL, "layers": TINY_LAYERS, "heads": TINY_HEADS}, "training_config": dataclasses.asdict(cfg)})
    ctx.register_artifact(manifest_path, "diagnostic", "Lab 29 data manifest, vocabulary, and controlled-training scope.")

    model, snapshots, train_rows, device = train_tiny_sequence(ctx, cfg, stoi, vocab)
    self_checks = run_tiny_self_checks(ctx, model, snapshots, tasks, device)
    behavior_rows, _caches = evaluate_checkpoints(ctx, model, snapshots, tasks, vocab, device)
    probe_rows, centroid_state = run_probe_selectivity(ctx, cfg, model, snapshots, stoi, device)
    feature_rows = write_feature_lineage(ctx, centroid_state)
    intervention_rows, intervention_depth, intervention_meta = run_intervention_transfer(ctx, model, snapshots, tasks, centroid_state, probe_rows, device)
    circuit_rows, events, counterexamples, metrics = summarize_circuits(ctx, behavior_rows, probe_rows, intervention_rows, tasks)
    evidence_rows = write_evidence_matrix(ctx, circuit_rows, events, metrics)
    save_state(ctx, cfg, vocab, snapshots, centroid_state, intervention_depth, intervention_meta)

    metrics = {**metrics, "data": data_info, "training_config": dataclasses.asdict(cfg), "intervention": intervention_meta, "self_checks_ok": self_checks.get("ok"), "evidence_components": len(evidence_rows)}
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 29 metrics, thresholds, and event steps.")
    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, circuit_rows)
    ctx.register_artifact(results_path, "table", "Alias of tables/checkpoint_circuit_summary.csv for dashboard tooling.")
    write_status_files(ctx, data_info, self_checks, metrics)
    write_cards(ctx, cfg, data_info, metrics, circuit_rows, events, counterexamples, self_checks)
    write_claims(ctx, events, circuit_rows, data_info)
    write_plots(ctx, circuit_rows, probe_rows, feature_rows, intervention_rows, behavior_rows, train_rows)
    print(f"[lab29] wrote {len(circuit_rows)} checkpoint summaries, {len(events)} events, and {len(counterexamples)} counterexamples")
