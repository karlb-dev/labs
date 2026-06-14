"""Lab 29: Training dynamics and circuit birth.

The default implementation is a controlled in-course time-lapse. It trains a
tiny causal transformer on synthetic induction-copy sequences, saves checkpoints
through training, and audits when behavior, decodability, attention motifs, and
intervention transfer appear. This keeps Tier A fast and reproducible while
teaching the checkpoint logic needed for Pythia or fine-tuning extensions.

Evidence level: OBS + DECODE + ATTR, with a small CAUSAL handle for activation
addition on the toy model. Claims are scoped to the controlled tiny sequence.
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
PROMPT_SET_CAPS = {"small": 11, "medium": 11, "full": 0}

INDUCTION_TOKENS = (
    "red", "blue", "green", "yellow", "orange", "purple",
    "silver", "bronze", "cyan", "magenta", "amber", "teal", "violet",
)
CONTROL_TOKENS = (
    "capital", "France", "Paris", "Rome", "language", "Italy",
    "Italian", "German", "month", "after", "January", "February", "March",
)
TRAIN_STEPS_BY_PROMPT_SET = {"small": 180, "medium": 260, "full": 420}
CHECKPOINT_FRACTIONS = (0.0, 0.03, 0.10, 0.25, 0.50, 1.0)
TINY_D_MODEL = 64
TINY_LAYERS = 2
TINY_HEADS = 4
TINY_MAX_LEN = 8
TINY_BATCH = 128
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
INTERVENTION_SCALE = 1.0


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


@dataclasses.dataclass
class ProbeExample:
    prompt_ids: list[int]
    target_id: int
    split: str


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


def safe_max(values: Sequence[Any], default: float = float("nan")) -> float:
    vals: list[float] = []
    for value in values:
        try:
            f = float(value)
        except Exception:
            continue
        if math.isfinite(f):
            vals.append(f)
    return max(vals) if vals else default


def data_path(args: Any) -> pathlib.Path:
    prompt_set = str(getattr(args, "prompt_set", "") or "")
    candidate = pathlib.Path(prompt_set)
    if prompt_set not in PROMPT_SET_CAPS and candidate.suffix.lower() in {".csv", ".tsv"}:
        return candidate if candidate.is_absolute() else (bench.COURSE_ROOT / candidate).resolve()
    return bench.COURSE_ROOT / "data" / DATA_FILE


def load_tasks(ctx: bench.RunContext) -> tuple[list[DynamicsTask], dict[str, Any]]:
    path = data_path(ctx.args)
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    tasks = [DynamicsTask(**row) for row in rows]
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
        "families": dict(Counter(t.task_family for t in tasks)),
        "splits": dict(Counter(t.split for t in tasks)),
        "science_ready": True,
        "science_scope": "controlled tiny-transformer training sequence, not a claim about external pretrained checkpoints",
    }
    return tasks, info


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


def tokenization_gate(ctx: bench.RunContext, tasks: list[DynamicsTask], stoi: Mapping[str, int]) -> list[DynamicsTask]:
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
            problems.append("prompt_too_long")
        if target_id == distractor_id:
            problems.append("target_equals_distractor")
        if not problems:
            task.prompt_ids = prompt_ids
            task.target_id = target_id
            task.distractor_id = distractor_id
            kept.append(task)
        rows.append({
            "item_id": task.item_id,
            "task_family": task.task_family,
            "split": task.split,
            "prompt_len": len(prompt_ids),
            "target_id": target_id if target_id >= 0 else "",
            "distractor_id": distractor_id if distractor_id >= 0 else "",
            "kept": not problems,
            "problems": ";".join(problems),
        })
    path = ctx.path("diagnostics", "tokenization_gate.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic", "Tiny-tokenizer task audit for Lab 29.")
    if not kept:
        raise RuntimeError("Lab 29 tokenization gate dropped every task.")
    return kept


def checkpoint_steps(prompt_set: str) -> list[int]:
    total = TRAIN_STEPS_BY_PROMPT_SET.get(prompt_set, TRAIN_STEPS_BY_PROMPT_SET["small"])
    return sorted({int(round(total * frac)) for frac in CHECKPOINT_FRACTIONS} | {total})


def clone_state_dict(model: Any) -> dict[str, Any]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def previous_match_position(ids: Sequence[int]) -> int | None:
    if len(ids) < 2:
        return None
    query = ids[-1]
    for pos in range(len(ids) - 2, -1, -1):
        if ids[pos] == query:
            return pos
    return None


def make_causal_mask(seq_len: int, device: Any) -> Any:
    import torch

    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


class TinyBlock:
    pass


def build_tiny_classes() -> tuple[type, type]:
    import torch
    import torch.nn as nn

    class _TinyBlock(nn.Module):
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
                y,
                y,
                y,
                attn_mask=mask,
                need_weights=return_attn,
                average_attn_weights=False,
            )
            x = x + attn_out
            x = x + self.mlp(self.ln2(x))
            return x, attn_weights if return_attn else None

    class _TinyTransformer(nn.Module):
        def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, max_len: int) -> None:
            super().__init__()
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.n_layers = n_layers
            self.max_len = max_len
            self.token_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_len, d_model)
            self.blocks = nn.ModuleList([_TinyBlock(d_model, n_heads) for _ in range(n_layers)])
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

    return _TinyBlock, _TinyTransformer


def make_induction_batch(
    stoi: Mapping[str, int],
    batch_size: int,
    generator: Any,
    device: Any,
) -> tuple[Any, Any]:
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
    stoi: Mapping[str, int],
    vocab: Sequence[str],
) -> tuple[Any, dict[int, dict[str, Any]], list[dict[str, Any]], Any]:
    import torch
    import torch.nn.functional as F

    _, TinyTransformerClass = build_tiny_classes()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(ctx.args.seed))
    model = TinyTransformerClass(len(vocab), TINY_D_MODEL, TINY_LAYERS, TINY_HEADS, TINY_MAX_LEN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps = checkpoint_steps(str(ctx.args.prompt_set))
    final_step = max(steps)
    gen = torch.Generator(device="cpu").manual_seed(int(ctx.args.seed) + 29029)
    snapshots: dict[int, dict[str, Any]] = {0: clone_state_dict(model)}
    train_rows: list[dict[str, Any]] = [{"step": 0, "loss": "", "train_accuracy": ""}]
    report_every = max(1, final_step // 5)
    for step in range(1, final_step + 1):
        model.train()
        x, y = make_induction_batch(stoi, TINY_BATCH, gen, device)
        logits, _, _ = model(x)
        loss = F.cross_entropy(logits[:, -1, :], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in steps or step % report_every == 0:
            with torch.no_grad():
                pred = logits[:, -1, :].argmax(dim=-1)
                acc = float((pred == y).float().mean().detach().cpu())
            train_rows.append({"step": step, "loss": rounded(float(loss.detach().cpu())), "train_accuracy": rounded(acc)})
            print(f"[lab29] tiny training step {step}/{final_step} loss={float(loss.detach().cpu()):.4f} acc={acc:.3f}")
        if step in steps:
            snapshots[step] = clone_state_dict(model)
    path = ctx.path("tables", "tiny_training_log.csv")
    bench.write_csv_with_context(ctx, path, train_rows)
    ctx.register_artifact(path, "table", "Tiny transformer training loss/accuracy log.")
    return model, snapshots, train_rows, device


def run_model_on_prompt(model: Any, ids: Sequence[int], device: Any, *, return_cache: bool = True, addition: tuple[int, Any, float] | None = None) -> tuple[Any, list[Any], list[Any]]:
    import torch

    x = torch.tensor([list(ids)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, streams, attentions = model(x, return_cache=return_cache, addition=addition)
    return logits[0, -1, :].detach().float().cpu(), streams, attentions


def logit_margin(logits: Any, target_id: int, distractor_id: int) -> float:
    return float(logits[target_id] - logits[distractor_id])


def logit_lens_event_depth(model: Any, streams: Sequence[Any], target_id: int, distractor_id: int) -> tuple[int | None, list[float]]:
    margins: list[float] = []
    for stream in streams:
        h = stream[0, -1, :].to(next(model.parameters()).device)
        with __import__("torch").no_grad():
            logits = model.unembed(model.final_ln(h))
        margins.append(float((logits[target_id] - logits[distractor_id]).detach().cpu()))
    event = next((i for i, margin in enumerate(margins) if margin > 0.0), None)
    return event, margins


def motif_score(attentions: Sequence[Any], ids: Sequence[int]) -> tuple[float, int | None, int | None, int | None]:
    prev = previous_match_position(ids)
    if prev is None or not attentions:
        return float("nan"), None, None, prev
    best_score = -float("inf")
    best_layer: int | None = None
    best_head: int | None = None
    final_pos = len(ids) - 1
    for layer, attn in enumerate(attentions, start=1):
        if attn is None:
            continue
        # [batch, heads, tgt, src]
        scores = attn[0, :, final_pos, prev]
        value, idx = scores.max(dim=0)
        if float(value) > best_score:
            best_score = float(value)
            best_layer = layer
            best_head = int(idx)
    return best_score, best_layer, best_head, prev


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
            motif, layer, head, prev = motif_score(attentions, task.prompt_ids)
            row = {
                "checkpoint_step": step,
                "item_id": task.item_id,
                "task_family": task.task_family,
                "split": task.split,
                "target": task.target,
                "distractor": task.distractor,
                "target_minus_distractor": rounded(margin),
                "correct": margin > 0.0,
                "top_token": vocab[int(logits.argmax())],
                "logit_lens_event_depth": "" if lens_event is None else lens_event,
                "final_depth_lens_margin": rounded(lens_margins[-1] if lens_margins else float("nan")),
                "best_prev_match_attention": rounded(motif),
                "best_motif_layer": "" if layer is None else layer,
                "best_motif_head": "" if head is None else head,
                "previous_match_position": "" if prev is None else prev,
            }
            rows.append(row)
            task_cache[task.item_id] = {"logits": logits, "streams": streams, "attentions": attentions, "row": row}
        caches[step] = task_cache
    path = ctx.path("tables", "checkpoint_behavior.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Checkpoint behavior, logit-lens, and motif measurements.")
    return rows, caches


def generate_probe_examples(stoi: Mapping[str, int], *, n_train: int = 120, n_eval: int = 72, seed: int = 0) -> list[ProbeExample]:
    import torch

    gen = torch.Generator(device="cpu").manual_seed(seed + 292929)
    color_ids = [stoi[tok] for tok in INDUCTION_TOKENS]
    examples: list[ProbeExample] = []
    for split, n in (("probe_train", n_train), ("probe_eval", n_eval)):
        for _ in range(n):
            perm = torch.randperm(len(color_ids), generator=gen)
            a, b, c = [color_ids[int(i)] for i in perm[:3]]
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


def centroid_probe(hidden: Any, labels: Sequence[int], splits: Sequence[str]) -> tuple[float, float, float, dict[int, Any]]:
    import torch
    import torch.nn.functional as F

    train_idx = [i for i, split in enumerate(splits) if split == "probe_train"]
    eval_idx = [i for i, split in enumerate(splits) if split == "probe_eval"]
    train_labels = [labels[i] for i in train_idx]
    classes = sorted(set(train_labels))
    centroids: dict[int, Any] = {}
    for cls in classes:
        idx = [i for i in train_idx if labels[i] == cls]
        centroids[cls] = hidden[idx].mean(dim=0)
    if not eval_idx or not centroids:
        return float("nan"), float("nan"), float("nan"), centroids

    centroid_mat = torch.stack([F.normalize(centroids[cls].float(), dim=0) for cls in classes], dim=0)
    eval_h = F.normalize(hidden[eval_idx].float(), dim=1)
    pred = centroid_mat.matmul(eval_h.T).argmax(dim=0)
    pred_labels = [classes[int(i)] for i in pred]
    acc = sum(1 for p, idx in zip(pred_labels, eval_idx) if p == labels[idx]) / len(eval_idx)

    shifted = train_labels[1:] + train_labels[:1]
    shuffled_centroids: dict[int, Any] = {}
    for cls in classes:
        idx = [train_idx[i] for i, lab in enumerate(shifted) if lab == cls]
        if idx:
            shuffled_centroids[cls] = hidden[idx].mean(dim=0)
    shuffled_classes = sorted(shuffled_centroids)
    shuffled_mat = torch.stack([F.normalize(shuffled_centroids[cls].float(), dim=0) for cls in shuffled_classes], dim=0)
    spred = shuffled_mat.matmul(eval_h.T).argmax(dim=0)
    spred_labels = [shuffled_classes[int(i)] for i in spred]
    shuffled_acc = sum(1 for p, idx in zip(spred_labels, eval_idx) if p == labels[idx]) / len(eval_idx)
    return acc, shuffled_acc, acc - shuffled_acc, centroids


def run_probe_selectivity(
    ctx: bench.RunContext,
    model: Any,
    snapshots: Mapping[int, Mapping[str, Any]],
    stoi: Mapping[str, int],
    device: Any,
) -> tuple[list[dict[str, Any]], dict[int, dict[int, dict[int, Any]]]]:
    examples = generate_probe_examples(stoi, seed=int(ctx.args.seed))
    rows: list[dict[str, Any]] = []
    centroid_state: dict[int, dict[int, dict[int, Any]]] = {}
    for step in sorted(snapshots):
        model.load_state_dict(snapshots[step])
        model.eval()
        hidden_by_depth = collect_hidden_by_depth(model, examples, device)
        centroid_state[step] = {}
        for depth, (hidden, labels, splits) in hidden_by_depth.items():
            acc, shuffled_acc, selectivity, centroids = centroid_probe(hidden, labels, splits)
            centroid_state[step][depth] = centroids
            rows.append({
                "checkpoint_step": step,
                "depth": depth,
                "probe_acc": rounded(acc),
                "shuffled_label_acc": rounded(shuffled_acc),
                "selectivity": rounded(selectivity),
                "n_probe_train": sum(1 for split in splits if split == "probe_train"),
                "n_probe_eval": sum(1 for split in splits if split == "probe_eval"),
            })
    path = ctx.path("tables", "checkpoint_probe_selectivity.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Centroid-probe decodability and shuffled-label controls by checkpoint/depth.")
    return rows, centroid_state


def cosine(a: Any, b: Any) -> float:
    import torch.nn.functional as F

    return float(F.cosine_similarity(a.float(), b.float(), dim=0).detach().cpu())


def write_feature_lineage(
    ctx: bench.RunContext,
    centroid_state: Mapping[int, Mapping[int, Mapping[int, Any]]],
) -> list[dict[str, Any]]:
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


def run_intervention_transfer(
    ctx: bench.RunContext,
    model: Any,
    snapshots: Mapping[int, Mapping[str, Any]],
    tasks: Sequence[DynamicsTask],
    centroid_state: Mapping[int, Mapping[int, Mapping[int, Any]]],
    probe_rows: Sequence[Mapping[str, Any]],
    device: Any,
) -> tuple[list[dict[str, Any]], int]:
    final_step = max(snapshots)
    final_probe_rows = [r for r in probe_rows if int(r["checkpoint_step"]) == final_step]
    best_depth = int(max(final_probe_rows, key=lambda r: float(r["selectivity"]))["depth"])
    final_centroids = centroid_state[final_step][best_depth]
    rows: list[dict[str, Any]] = []
    for step in sorted(snapshots):
        model.load_state_dict(snapshots[step])
        model.eval()
        for task in tasks:
            if task.task_family != "induction_copy":
                continue
            if task.target_id not in final_centroids or task.distractor_id not in final_centroids:
                continue
            direction = (final_centroids[task.target_id] - final_centroids[task.distractor_id]).float().cpu()
            rand = random_like(direction, f"{step}|{task.item_id}|intervention")
            base_logits, _, _ = run_model_on_prompt(model, task.prompt_ids, device, return_cache=False)
            edited_logits, _, _ = run_model_on_prompt(
                model,
                task.prompt_ids,
                device,
                return_cache=False,
                addition=(best_depth, direction, INTERVENTION_SCALE),
            )
            random_logits, _, _ = run_model_on_prompt(
                model,
                task.prompt_ids,
                device,
                return_cache=False,
                addition=(best_depth, rand, INTERVENTION_SCALE),
            )
            base = logit_margin(base_logits, task.target_id, task.distractor_id)
            edited = logit_margin(edited_logits, task.target_id, task.distractor_id)
            random_margin = logit_margin(random_logits, task.target_id, task.distractor_id)
            rows.append({
                "checkpoint_step": step,
                "item_id": task.item_id,
                "split": task.split,
                "depth": best_depth,
                "base_margin": rounded(base),
                "edited_margin": rounded(edited),
                "random_direction_margin": rounded(random_margin),
                "intervention_gain": rounded(edited - base),
                "random_gain": rounded(random_margin - base),
                "control_gap": rounded((edited - base) - (random_margin - base)),
            })
    path = ctx.path("tables", "intervention_transfer.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Final-checkpoint centroid-direction activation additions transferred to earlier checkpoints.")
    return rows, best_depth


def summarize_circuits(
    ctx: bench.RunContext,
    behavior_rows: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    intervention_rows: Sequence[Mapping[str, Any]],
    tasks: Sequence[DynamicsTask],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prev_best_depth: int | None = None
    induction_count = sum(1 for t in tasks if t.task_family == "induction_copy")
    for step in sorted({int(r["checkpoint_step"]) for r in behavior_rows}):
        b_rows = [r for r in behavior_rows if int(r["checkpoint_step"]) == step and r["task_family"] == "induction_copy"]
        control_rows = [r for r in behavior_rows if int(r["checkpoint_step"]) == step and r["task_family"] != "induction_copy"]
        p_rows = [r for r in probe_rows if int(r["checkpoint_step"]) == step]
        i_rows = [r for r in intervention_rows if int(r["checkpoint_step"]) == step]
        best_probe = max(p_rows, key=lambda r: float(r["selectivity"]))
        motif_rows = [r for r in b_rows if r["best_prev_match_attention"] != ""]
        best_motif = max(motif_rows, key=lambda r: float(r["best_prev_match_attention"])) if motif_rows else {}
        random_attention = safe_mean([1.0 / len(t.prompt_ids) for t in tasks if t.task_family == "induction_copy"])
        best_motif_score = float(best_motif.get("best_prev_match_attention", float("nan"))) if best_motif else float("nan")
        mean_motif_score = safe_mean([r["best_prev_match_attention"] for r in motif_rows])
        motif_gap = mean_motif_score - random_attention if math.isfinite(mean_motif_score) else float("nan")
        behavior_acc = safe_mean([1.0 if r["correct"] else 0.0 for r in b_rows], default=0.0)
        control_acc = safe_mean([1.0 if r["correct"] else 0.0 for r in control_rows], default=0.0)
        intervention_gain = safe_mean([r["intervention_gain"] for r in i_rows])
        random_gain = safe_mean([r["random_gain"] for r in i_rows])
        best_depth = int(best_probe["depth"])
        if behavior_acc < 0.75 and float(best_probe["probe_acc"]) < 0.75 and motif_gap < 0.15:
            phase = "absent_or_random"
        elif float(best_probe["probe_acc"]) >= 0.75 and behavior_acc < 0.75:
            phase = "decodable_before_behavioral"
        elif behavior_acc >= 0.75 and float(best_probe["probe_acc"]) < 0.75:
            phase = "behavioral_before_decodable"
        elif prev_best_depth is not None and best_depth != prev_best_depth and behavior_acc >= 0.75:
            phase = "migration"
        elif behavior_acc >= 0.75 and float(best_probe["probe_acc"]) >= 0.75 and motif_gap >= 0.15:
            phase = "circuit_present"
        elif behavior_acc >= 0.75 and float(best_probe["probe_acc"]) >= 0.75:
            phase = "behavioral_decodable_no_mean_attention_motif"
        else:
            phase = "sharpening_or_redistributed"
        rows.append({
            "checkpoint_step": step,
            "induction_accuracy": rounded(behavior_acc),
            "control_accuracy": rounded(control_acc),
            "mean_induction_margin": rounded(safe_mean([r["target_minus_distractor"] for r in b_rows])),
            "best_probe_depth": best_depth,
            "best_probe_acc": best_probe["probe_acc"],
            "best_probe_selectivity": best_probe["selectivity"],
            "best_motif_attention": rounded(best_motif_score),
            "mean_prev_match_attention": rounded(mean_motif_score),
            "best_motif_layer": best_motif.get("best_motif_layer", ""),
            "best_motif_head": best_motif.get("best_motif_head", ""),
            "motif_control_gap": rounded(motif_gap),
            "mean_intervention_gain": rounded(intervention_gain),
            "mean_random_gain": rounded(random_gain),
            "intervention_control_gap": rounded(intervention_gain - random_gain),
            "phase": phase,
            "n_induction_tasks": induction_count,
        })
        prev_best_depth = best_depth
    event_specs = [
        ("behavior_emergence", lambda r: float(r["induction_accuracy"]) >= 0.75, "induction accuracy >= 0.75"),
        ("decodability_emergence", lambda r: float(r["best_probe_acc"]) >= 0.75 and float(r["best_probe_selectivity"]) >= 0.20, "probe acc >= 0.75 and selectivity >= 0.20"),
        ("motif_emergence", lambda r: float(r["motif_control_gap"]) >= 0.15, "mean previous-match attention gap >= 0.15"),
        (
            "intervention_transfer_emergence",
            lambda r: float(r["intervention_control_gap"]) >= 0.20
            and float(r["best_probe_selectivity"]) >= 0.20
            and float(r["induction_accuracy"]) >= 0.75,
            "centroid-direction gain beats random by >= 0.20 after behavior and probe selectivity are present",
        ),
    ]
    events: list[dict[str, Any]] = []
    for event, pred, threshold in event_specs:
        hits = [r for r in rows if pred(r)]
        first = hits[0] if hits else None
        events.append({
            "event": event,
            "first_checkpoint_step": "" if first is None else first["checkpoint_step"],
            "threshold": threshold,
            "observed": "" if first is None else json.dumps({k: first[k] for k in ("induction_accuracy", "best_probe_acc", "best_probe_selectivity", "motif_control_gap", "intervention_control_gap")}),
            "claim_status": "observed_in_controlled_sequence" if first is not None else "not_observed",
        })
    path = ctx.path("tables", "checkpoint_circuit_summary.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Checkpoint-level phase classification and circuit summary.")
    event_path = ctx.path("tables", "mechanism_birth_events.csv")
    bench.write_csv_with_context(ctx, event_path, events)
    ctx.register_artifact(event_path, "table", "Thresholded behavior/decoding/motif/intervention birth events.")
    metrics = {
        "n_checkpoints": len(rows),
        "n_induction_tasks": induction_count,
        "final_induction_accuracy": rows[-1]["induction_accuracy"] if rows else "",
        "final_probe_acc": rows[-1]["best_probe_acc"] if rows else "",
        "behavior_emergence_step": next((r["first_checkpoint_step"] for r in events if r["event"] == "behavior_emergence"), ""),
        "decodability_emergence_step": next((r["first_checkpoint_step"] for r in events if r["event"] == "decodability_emergence"), ""),
        "motif_emergence_step": next((r["first_checkpoint_step"] for r in events if r["event"] == "motif_emergence"), ""),
    }
    return rows, events, metrics


def save_state(
    ctx: bench.RunContext,
    vocab: Sequence[str],
    snapshots: Mapping[int, Mapping[str, Any]],
    centroid_state: Mapping[int, Mapping[int, Mapping[int, Any]]],
    intervention_depth: int,
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
        "state_dicts": snapshots,
        "centroids": centroid_state,
        "intervention_depth": intervention_depth,
        "note": "Controlled tiny-model checkpoints; do not compare these steps to pretrained-model training tokens.",
    }
    path = ctx.path("state", "checkpoint_directions.pt")
    torch.save(payload, path)
    ctx.register_artifact(path, "state", "Tiny checkpoint sequence, centroid directions, and intervention depth.")


def write_method_card(
    ctx: bench.RunContext,
    circuit_rows: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Lab 29 method card",
        "",
        "This lab uses a controlled tiny transformer trained during the run. It is a time-lapse microscope, not a Pythia result.",
        "",
        f"- training task: synthetic induction-copy sequences over {len(INDUCTION_TOKENS)} color tokens",
        f"- architecture: {TINY_LAYERS} layers, {TINY_HEADS} heads, d_model={TINY_D_MODEL}",
        "- negative controls: frozen relation/calendar rows that are never trained",
        "- decodability control: shuffled-label centroid probe",
        "- intervention: final-checkpoint centroid direction added to earlier checkpoints",
        "- evidence rung: `OBS + DECODE + ATTR`, plus scoped toy `CAUSAL` intervention transfer",
        "- forbidden claim: the model first learned a concept at exactly this step",
        "",
        "| event | first checkpoint | status |",
        "|---|---:|---|",
    ]
    for row in events:
        lines.append(f"| {row['event']} | {row['first_checkpoint_step']} | {row['claim_status']} |")
    lines += [
        "",
        "## Final checkpoint",
        "",
        "| step | induction acc | probe acc | motif gap | intervention gap | phase |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    if circuit_rows:
        row = circuit_rows[-1]
        lines.append(
            f"| {row['checkpoint_step']} | {row['induction_accuracy']} | {row['best_probe_acc']} | "
            f"{row['motif_control_gap']} | {row['intervention_control_gap']} | {row['phase']} |"
        )
    path = ctx.path("method_card.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 29 method card and event summary.")


def write_operationalization_audit(ctx: bench.RunContext, circuit_rows: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Lab 29 operationalization audit",
        "",
        "Favorite interpretation under attack: a circuit is born at a precise training step.",
        "",
        "## What the measurement can say",
        "",
        "The controlled checkpoint sequence shows when behavior, centroid decodability, previous-match attention, and a centroid-direction intervention cross preregistered thresholds.",
        "",
        "## What it cannot say",
        "",
        "It cannot identify the exact instant a concept was learned, and it cannot transfer claims from this tiny synthetic model to pretrained LLM checkpoints without rerunning the same audit there.",
        "",
        "## Cheap explanations",
        "",
        "- Behavior improves before the measured motif appears.",
        "- The probe is linearly decodable but behavior is absent.",
        "- Shuffled labels decode just as well.",
        "- A final-checkpoint intervention works only because residual axes stayed aligned by construction.",
        "- The untrained control task changes along with the trained task.",
        "",
        "## Phase trajectory",
        "",
    ]
    for row in circuit_rows:
        lines.append(
            f"- step `{row['checkpoint_step']}`: `{row['phase']}` "
            f"(behavior={row['induction_accuracy']}, probe={row['best_probe_acc']}, motif_gap={row['motif_control_gap']})."
        )
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Controls and non-claims for Lab 29 training dynamics.")


def write_run_summary(
    ctx: bench.RunContext,
    data_info: Mapping[str, Any],
    metrics: Mapping[str, Any],
    circuit_rows: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Lab 29 run summary: training dynamics and circuit birth",
        "",
        f"- data rows: {data_info['n_rows_selected']} selected from `{pathlib.Path(str(data_info['data_path'])).name}`",
        f"- science scope: {data_info['science_scope']}",
        f"- checkpoints: {metrics['n_checkpoints']}",
        f"- final induction accuracy: `{metrics['final_induction_accuracy']}`",
        f"- final probe accuracy: `{metrics['final_probe_acc']}`",
        "",
        "## Birth-event ledger",
        "",
        "| event | first checkpoint | status |",
        "|---|---:|---|",
    ]
    for row in events:
        lines.append(f"| {row['event']} | {row['first_checkpoint_step']} | {row['claim_status']} |")
    lines += [
        "",
        "## Reading order",
        "",
        "1. `method_card.md` for the controlled training scope.",
        "2. `tables/checkpoint_behavior.csv` for task margins and attention motifs.",
        "3. `tables/checkpoint_probe_selectivity.csv` for decodability and shuffled-label controls.",
        "4. `tables/checkpoint_circuit_summary.csv` for phase labels.",
        "5. `tables/feature_lineage.csv` and `tables/intervention_transfer.csv` for migration and causal-transfer caveats.",
        "",
        "## Smallest surviving claim",
        "",
        "In this controlled tiny sequence, thresholded behavior, decodability, motif, and intervention-transfer events can be ordered. The steps are measurement thresholds, not exact learning instants.",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 29 run summary and reading order.")


def write_plot_guide(ctx: bench.RunContext) -> None:
    rows = [
        {"plot": "plots/training_dynamics_dashboard.png", "read_for": "Behavior, probe, motif, and intervention timelines.", "non_claim": "Threshold crossings are not exact learning instants."},
        {"plot": "plots/behavior_vs_decodability_timeline.png", "read_for": "Behavior vs centroid-probe emergence.", "non_claim": "Probe decodability is not behavior."},
        {"plot": "plots/circuit_birth_atlas.png", "read_for": "Phase labels over checkpoints.", "non_claim": "Phase labels depend on thresholds."},
        {"plot": "plots/depth_migration_map.png", "read_for": "Best probe depth migration.", "non_claim": "Depth migration is a summary, not a unique feature path."},
        {"plot": "plots/checkpoint_feature_lineage.png", "read_for": "Centroid similarity to final checkpoint.", "non_claim": "Cosine stability is not identity."},
        {"plot": "plots/intervention_transfer_over_time.png", "read_for": "Final-direction intervention transfer.", "non_claim": "Tiny-model axis alignment is easier than cross-model transfer."},
        {"plot": "plots/random_model_control_panel.png", "read_for": "Checkpoint-zero and untrained-control behavior.", "non_claim": "Random controls are a floor, not a full null distribution."},
    ]
    path = ctx.path("plots", "plot_reading_guide.csv")
    bench.write_csv(path, rows)
    ctx.register_artifact(path, "table", "Plot reading guide for Lab 29.")


def write_plots(
    ctx: bench.RunContext,
    circuit_rows: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    intervention_rows: Sequence[Mapping[str, Any]],
    behavior_rows: Sequence[Mapping[str, Any]],
) -> None:
    write_plot_guide(ctx)
    if ctx.args.no_plots:
        return
    import matplotlib.pyplot as plt
    import numpy as np

    steps = [int(r["checkpoint_step"]) for r in circuit_rows]
    behavior = [float(r["induction_accuracy"]) for r in circuit_rows]
    probe = [float(r["best_probe_acc"]) for r in circuit_rows]
    motif = [float(r["motif_control_gap"]) for r in circuit_rows]
    intervention = [float(r["intervention_control_gap"]) for r in circuit_rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Lab 29 training dynamics dashboard", fontsize=14, fontweight="bold")
    axes[0, 0].plot(steps, behavior, marker="o", label="behavior", color="#0072B2")
    axes[0, 0].plot(steps, probe, marker="o", label="probe", color="#009E73")
    axes[0, 0].axhline(0.75, color="#444444", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("Behavior vs decodability")
    axes[0, 0].set_xlabel("training step")
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(steps, motif, marker="o", color="#CC79A7")
    axes[0, 1].axhline(0.15, color="#444444", linestyle="--", linewidth=0.8)
    axes[0, 1].set_title("Previous-match attention gap")
    axes[0, 1].set_xlabel("training step")

    axes[1, 0].plot(steps, intervention, marker="o", color="#D55E00")
    axes[1, 0].axhline(0.20, color="#444444", linestyle="--", linewidth=0.8)
    axes[1, 0].set_title("Intervention transfer gap")
    axes[1, 0].set_xlabel("training step")

    phase_ids = {phase: i for i, phase in enumerate(sorted({str(r["phase"]) for r in circuit_rows}))}
    axes[1, 1].step(steps, [phase_ids[str(r["phase"])] for r in circuit_rows], where="mid", color="#E69F00")
    axes[1, 1].set_yticks(list(phase_ids.values()), list(phase_ids.keys()), fontsize=7)
    axes[1, 1].set_title("Phase trajectory")
    axes[1, 1].set_xlabel("training step")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    bench.save_figure(ctx, fig, "training_dynamics_dashboard.png", "Lab 29 training dynamics dashboard.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, behavior, marker="o", label="behavior", color="#0072B2")
    ax.plot(steps, probe, marker="o", label="decodability", color="#009E73")
    ax.axhline(0.75, color="#444444", linestyle="--", linewidth=0.8)
    ax.set_xlabel("training step")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Behavior vs decodability timeline")
    ax.legend(frameon=False)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "behavior_vs_decodability_timeline.png", "Behavior/decodability timeline.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.imshow(np.array([[float(r["induction_accuracy"]), float(r["best_probe_acc"]), float(r["motif_control_gap"]), float(r["intervention_control_gap"])] for r in circuit_rows]), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(steps)), steps)
    ax.set_xticks(range(4), ["behavior", "probe", "motif", "intervention"], rotation=20)
    ax.set_title("Circuit birth atlas")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "circuit_birth_atlas.png", "Circuit birth metric atlas.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, [int(r["best_probe_depth"]) for r in circuit_rows], marker="o", color="#0072B2")
    ax.set_xlabel("training step")
    ax.set_ylabel("best probe depth")
    ax.set_title("Depth migration map")
    ax.set_yticks(range(TINY_LAYERS + 1))
    fig.tight_layout()
    bench.save_figure(ctx, fig, "depth_migration_map.png", "Best decodability depth over checkpoints.")

    final_like = [r for r in feature_rows if int(r["depth"]) == int(r["best_matching_final_depth"] or -1)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for depth in sorted({int(r["depth"]) for r in feature_rows}):
        rows = [r for r in feature_rows if int(r["depth"]) == depth]
        ax.plot([int(r["checkpoint_step"]) for r in rows], [float(r["same_depth_cosine_to_final"]) for r in rows], marker="o", label=f"depth {depth}")
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
        gains_by_step[int(row["checkpoint_step"])].append(float(row["intervention_gain"]))
        random_by_step[int(row["checkpoint_step"])].append(float(row["random_gain"]))
    ax.plot(steps, [safe_mean(gains_by_step[s]) for s in steps], marker="o", label="final direction", color="#D55E00")
    ax.plot(steps, [safe_mean(random_by_step[s]) for s in steps], marker="o", label="random", color="#999999")
    ax.set_xlabel("training step")
    ax.set_ylabel("margin gain")
    ax.set_title("Intervention transfer over time")
    ax.legend(frameon=False)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "intervention_transfer_over_time.png", "Final-direction intervention transfer over checkpoints.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    control = [r for r in behavior_rows if r["task_family"] != "induction_copy"]
    control_steps = sorted({int(r["checkpoint_step"]) for r in control})
    ax.plot(control_steps, [safe_mean([1.0 if r["correct"] else 0.0 for r in control if int(r["checkpoint_step"]) == s]) for s in control_steps], marker="o", label="untrained controls", color="#999999")
    ax.plot(steps, behavior, marker="o", label="trained induction", color="#0072B2")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("training step")
    ax.set_ylabel("accuracy")
    ax.set_title("Random/untrained control panel")
    ax.legend(frameon=False)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "random_model_control_panel.png", "Random-model and untrained-control panel.")


def write_claims(ctx: bench.RunContext, events: Sequence[Mapping[str, Any]], circuit_rows: Sequence[Mapping[str, Any]]) -> None:
    run_name = ctx.run_dir.name
    final = circuit_rows[-1] if circuit_rows else {}
    claims = []
    for i, row in enumerate(events, start=1):
        claims.append({
            "id": f"{LAB_ID}-C{i}",
            "tag": "OBS,DECODE,ATTR" if row["event"] != "intervention_transfer_emergence" else "OBS,DECODE,CAUSAL",
            "text": (
                f"In the controlled tiny checkpoint sequence, `{row['event']}` first crossed the lab threshold at "
                f"checkpoint `{row['first_checkpoint_step']}`. Status: {row['claim_status']}. "
                f"Final phase: {final.get('phase', '')}."
            ),
            "artifact": f"runs/{run_name}/tables/mechanism_birth_events.csv",
            "falsifier": "A shuffled-label probe, untrained control task, or random-direction intervention crosses the same threshold.",
        })
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    tasks, data_info = load_tasks(ctx)
    stoi, vocab = build_vocab(tasks)
    tasks = tokenization_gate(ctx, tasks, stoi)
    manifest_path = ctx.path("diagnostics", "data_manifest.json")
    bench.write_json(manifest_path, {**data_info, "vocab_size": len(vocab), "tiny_model": {"d_model": TINY_D_MODEL, "layers": TINY_LAYERS, "heads": TINY_HEADS}})
    ctx.register_artifact(manifest_path, "diagnostic", "Lab 29 data manifest and controlled-training scope.")
    bench.run_hook_parity_check(ctx, bundle, tasks[0].prompt)
    first = bench.run_with_residual_cache(bundle, tasks[0].prompt)
    bench.run_lens_self_check(ctx, bundle, first)
    bench.run_patch_noop_check(ctx, bundle, tasks[0].prompt)
    model, snapshots, train_rows, device = train_tiny_sequence(ctx, stoi, vocab)
    behavior_rows, caches = evaluate_checkpoints(ctx, model, snapshots, tasks, vocab, device)
    probe_rows, centroid_state = run_probe_selectivity(ctx, model, snapshots, stoi, device)
    feature_rows = write_feature_lineage(ctx, centroid_state)
    intervention_rows, intervention_depth = run_intervention_transfer(ctx, model, snapshots, tasks, centroid_state, probe_rows, device)
    circuit_rows, events, metrics = summarize_circuits(ctx, behavior_rows, probe_rows, intervention_rows, tasks)
    save_state(ctx, vocab, snapshots, centroid_state, intervention_depth)
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, {**metrics, "data": data_info, "training_steps": sorted(snapshots)})
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 29 metrics.")
    write_method_card(ctx, circuit_rows, events)
    write_operationalization_audit(ctx, circuit_rows, events)
    write_run_summary(ctx, data_info, metrics, circuit_rows, events)
    write_claims(ctx, events, circuit_rows)
    write_plots(ctx, circuit_rows, probe_rows, feature_rows, intervention_rows, behavior_rows)
    print(f"[lab29] wrote {len(circuit_rows)} checkpoint summaries and {len(events)} birth events")
