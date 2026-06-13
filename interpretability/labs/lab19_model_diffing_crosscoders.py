"""Lab 19: model diffing with a paired sparse crosscoder.

This lab asks a deliberately narrow version of a tempting question:

    When a base model becomes an instruct model, what representational handles
    appear, disappear, or change shape under a matched prompt distribution?

The unit of evidence is a paired feature dictionary trained on matched
residual-stream activations from two models. A feature taxonomy is DECODE/ATTR
unless it survives the cheap-explanation audit. Optional activation addition on
benign prompts can add a small CAUSAL handle claim, but it is not a mechanism
claim by itself.

The lab is written to be inspectable rather than maximal. It supports the full
science pair, OLMo base versus OLMo instruct, and also an identity-pair smoke
run where model A and model B are intentionally the same. An identity-pair run
should find mostly shared or noisy features. If it finds a bright forest of
"model B only" features, the microscope is fogged.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

import interp_bench as bench

LAB_ID = "L19"

DEFAULT_COMPARE_MODEL = "allenai/Olmo-3-7B-Instruct"
DEFAULT_SYSTEM = "You are a careful assistant. Keep answers concise and accurate."

CROSSCODER_FEATURES = 128
TRAIN_STEPS_TIER_A = 180
TRAIN_STEPS_TIER_B = 420
LEARNING_RATE = 2e-3
L1_WEIGHT = 1.8e-3
SIDE_RECON_WEIGHT = 0.35
WEIGHT_DECAY = 1e-5
EVAL_FRACTION = 0.25

GALLERY_FEATURES = 24
GALLERY_CONTEXTS = 6
RANDOM_BASELINE_DRAWS = 256
MAX_NEW_TOKENS = 48
ENGINE_MAX_CONCURRENT = 8
STEERING_DOSE = 0.70

VOICE_MARKERS = (
    "i can", "i'll", "i will", "happy to", "glad to", "help", "assist",
    "here's", "here are", "sure", "certainly", "let's", "step", "please",
)
POLITENESS_MARKERS = (
    "please", "thanks", "thank you", "happy to", "glad", "appreciate", "polite",
)
HEDGING_MARKERS = (
    "might", "may", "could", "likely", "uncertain", "not sure", "depends", "check",
    "verify", "I don't know", "I cannot determine".lower(),
)
REFUSAL_MARKERS = (
    "can't help", "cannot help", "i can't", "i cannot", "unable to", "not able to",
    "private", "sensitive", "sorry",
)
SELF_SITUATION_MARKERS = (
    "as an ai", "language model", "assistant", "i don't have", "i do not have",
)
DISCLOSURE_MARKERS = (
    "i should be transparent", "to be clear", "i don't know", "i do not know",
    "verify", "source", "evidence",
)


@dataclasses.dataclass
class PromptItem:
    prompt_id: str
    prompt_group: str
    family: str
    source: str
    variant: str
    text: str
    user_message: str = ""
    note: str = ""


@dataclasses.dataclass
class PairActivations:
    prompt_rows: list[dict[str, Any]]
    x_a: Any
    x_b: Any
    depth_a: int
    depth_b: int
    split: dict[str, list[int]]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def stable_hash_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rounded(x: Any, ndigits: int = 4) -> Any:
    try:
        xf = float(x)
    except Exception:
        return x
    if not math.isfinite(xf):
        return None
    return round(xf, ndigits)


def safe_fmean(vals: Iterable[Any], default: float = float("nan")) -> float:
    finite = []
    for value in vals:
        try:
            f = float(value)
        except Exception:
            continue
        if math.isfinite(f):
            finite.append(f)
    return float(statistics.fmean(finite)) if finite else default


def safe_stdev(vals: Iterable[Any], default: float = float("nan")) -> float:
    finite = []
    for value in vals:
        try:
            f = float(value)
        except Exception:
            continue
        if math.isfinite(f):
            finite.append(f)
    if len(finite) < 2:
        return default
    return float(statistics.stdev(finite))


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = []
    for x, y in zip(xs, ys):
        try:
            xf = float(x)
            yf = float(y)
        except Exception:
            continue
        if math.isfinite(xf) and math.isfinite(yf):
            pairs.append((xf, yf))
    if len(pairs) < 3:
        return float("nan")
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    mx = statistics.fmean(xvals)
    my = statistics.fmean(yvals)
    dx = math.sqrt(sum((x - mx) ** 2 for x in xvals))
    dy = math.sqrt(sum((y - my) ** 2 for y in yvals))
    if dx < 1e-12 or dy < 1e-12:
        return float("nan")
    return float(sum((x - mx) * (y - my) for x, y in pairs) / (dx * dy))


def cosine(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or not xs:
        return float("nan")
    dot = sum(float(x) * float(y) for x, y in zip(xs, ys))
    nx = math.sqrt(sum(float(x) * float(x) for x in xs))
    ny = math.sqrt(sum(float(y) * float(y) for y in ys))
    if nx < 1e-12 or ny < 1e-12:
        return float("nan")
    return float(dot / (nx * ny))


def data_path(name: str) -> pathlib.Path:
    return bench.COURSE_ROOT / "data" / name


def decode_cell(text: Any) -> str:
    return str(text or "").replace("\\n", "\n")


def marker_any(text: str, markers: Sequence[str]) -> float:
    low = text.lower()
    return 1.0 if any(marker.lower() in low for marker in markers) else 0.0


def marker_count(text: str, markers: Sequence[str]) -> int:
    low = text.lower()
    return sum(low.count(marker.lower()) for marker in markers)


def repetition_rate(text: str) -> float:
    toks = [t.strip(".,;:!?()[]{}\"'").lower() for t in text.split()]
    toks = [t for t in toks if t]
    if not toks:
        return 0.0
    return 1.0 - (len(set(toks)) / len(toks))


def infer_model_role(model_id: str, *, is_compare: bool) -> str:
    forced = os.environ.get("LAB19_ROLE_B" if is_compare else "LAB19_ROLE_A")
    if forced:
        return forced.strip().lower().replace(" ", "_")
    low = model_id.lower()
    if any(tag in low for tag in ("instruct", "chat", "sft", "dpo", "ppo", "think")):
        return "instruct"
    if any(tag in low for tag in ("base", "pt", "pretrain")):
        return "base"
    return "model_b" if is_compare else "model_a"


def generic_to_role_taxonomy(generic: str, role_a: str, role_b: str) -> str:
    if generic == "model_a_only":
        return f"{role_a}_only"
    if generic == "model_b_only":
        return f"{role_b}_only"
    return generic


def token_ids(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False).get("input_ids", [])
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def render_chat_with_tokenizer(tokenizer: Any, user_message: str, *, system: str = DEFAULT_SYSTEM) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]
    template = getattr(tokenizer, "chat_template", None)
    if template:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False)
    return f"System: {system}\nUser: {user_message}\nAssistant:"


def render_for_bundle(bundle: bench.ModelBundle, user_message: str, *, system: str = DEFAULT_SYSTEM) -> str:
    return render_chat_with_tokenizer(bundle.tokenizer, user_message, system=system)


def add_prompt(
    items: list[PromptItem],
    prompt_id: str,
    prompt_group: str,
    family: str,
    source: str,
    variant: str,
    text: str,
    *,
    user_message: str = "",
    note: str = "",
) -> None:
    text = decode_cell(text).strip()
    if not text:
        return
    group = prompt_group or prompt_id
    items.append(PromptItem(prompt_id, group, family, source, variant, text, user_message, note))


# ---------------------------------------------------------------------------
# Prompt inventory
# ---------------------------------------------------------------------------


def comparison_tokenizer(model_id: str, revision: str | None, ctx: bench.RunContext) -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=ctx.args.trust_remote_code,
        local_files_only=ctx.args.local_files_only,
    )
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token", None):
        tok.pad_token = tok.eos_token
    return tok


def builtin_prompts(compare_tok: Any) -> list[PromptItem]:
    items: list[PromptItem] = []
    rows = [
        ("assistant_notes", "assistant_voice", "Please help me organize my project notes into next steps."),
        ("assistant_uncertainty", "assistant_voice", "I am unsure whether a factual claim is true. How should I answer carefully?"),
        ("assistant_polite_correction", "assistant_voice", "Write a polite correction to a teammate who misstated a date."),
        ("assistant_boundary", "assistant_voice", "A classmate asks for private records. What should I do?"),
        ("technical_python", "technical", "Explain why a Python function might return None unexpectedly."),
        ("technical_vector", "technical", "Explain cosine similarity to a new machine learning student."),
        ("factual_capital", "plain_factual", "The capital of Japan is"),
        ("factual_science", "plain_factual", "Water freezes at"),
        ("style_plain", "style_control", "Rewrite this sentence to be shorter: the committee will meet next Tuesday."),
        ("style_formal", "style_control", "Rewrite this sentence in a more formal register: thanks for the help."),
    ]
    for pid, family, user_message in rows:
        add_prompt(items, pid + "__raw", pid, family, "authored_lab19", "raw", user_message, user_message=user_message)
        add_prompt(
            items,
            pid + "__compare_chat",
            pid,
            family,
            "authored_lab19",
            "compare_chat",
            render_chat_with_tokenizer(compare_tok, user_message),
            user_message=user_message,
        )
    return items


def load_custom_prompts(path: pathlib.Path, compare_tok: Any) -> list[PromptItem]:
    items: list[PromptItem] = []
    suffix = path.suffix.lower()
    records: list[dict[str, Any]] = []
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    elif suffix == ".json":
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            records = [dict(r) for r in payload]
        elif isinstance(payload, dict):
            records = [dict(r) for r in payload.get("prompts", [])]
    elif suffix in {".csv", ".tsv"}:
        with path.open(newline="", encoding="utf-8") as f:
            dialect = "excel-tab" if suffix == ".tsv" else "excel"
            records = [dict(r) for r in csv.DictReader(f, dialect=dialect)]
    else:
        raise RuntimeError(f"Unsupported Lab 19 prompt inventory format: {path}")

    for i, row in enumerate(records):
        pid = str(row.get("prompt_id") or row.get("id") or f"custom_{i:04d}")
        group = str(row.get("prompt_group") or row.get("group") or pid)
        family = str(row.get("family") or row.get("category") or "custom")
        variant = str(row.get("variant") or "raw")
        user_message = decode_cell(row.get("user_message") or row.get("message") or row.get("prompt") or row.get("text") or "")
        raw_text = decode_cell(row.get("text") or row.get("prompt") or user_message)
        if str(row.get("render_chat", "")).lower() in {"1", "true", "yes", "compare_chat"}:
            chat_text = render_chat_with_tokenizer(compare_tok, user_message or raw_text)
            add_prompt(items, pid + "__compare_chat", group, family, path.name, "compare_chat", chat_text, user_message=user_message or raw_text)
        else:
            add_prompt(items, pid, group, family, path.name, variant, raw_text, user_message=user_message or raw_text)
            if str(row.get("also_chat", "")).lower() in {"1", "true", "yes"}:
                chat_text = render_chat_with_tokenizer(compare_tok, user_message or raw_text)
                add_prompt(items, pid + "__compare_chat", group, family, path.name, "compare_chat", chat_text, user_message=user_message or raw_text)
    return items


def load_course_prompt_sources(compare_tok: Any) -> list[PromptItem]:
    """Pull a small, matched prompt inventory from earlier advanced labs when present."""

    items: list[PromptItem] = []

    persona_path = data_path("persona_register_pairs.csv")
    if persona_path.exists():
        with persona_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                item_id = row.get("item_id") or row.get("id") or sha256_text(str(row))[:8]
                trait = row.get("trait", "persona")
                eval_prompt = decode_cell(row.get("eval_prompt", ""))
                pos_prompt = decode_cell(row.get("prompt_positive", ""))
                for suffix, user_message in (("eval", eval_prompt), ("positive", pos_prompt)):
                    if user_message:
                        group = f"persona_{item_id}_{suffix}"
                        add_prompt(items, f"{group}__raw", group, trait, persona_path.name, "raw", user_message, user_message=user_message)
                        add_prompt(items, f"{group}__chat", group, trait, persona_path.name, "compare_chat", render_chat_with_tokenizer(compare_tok, user_message), user_message=user_message)

    syc_path = data_path("sycophancy_pressure_items.csv")
    if syc_path.exists():
        with syc_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rid = row.get("row_id") or row.get("item_id") or sha256_text(str(row))[:8]
                cond = row.get("condition", "unknown")
                user_message = decode_cell(row.get("user_message") or row.get("prompt") or "")
                if user_message:
                    group = f"sycophancy_{rid}"
                    fam = "sycophancy_" + cond
                    add_prompt(items, f"{group}__raw", group, fam, syc_path.name, "raw", user_message, user_message=user_message)
                    if cond in {"neutral", "false_belief", "mild_pressure", "identity_pressure"}:
                        add_prompt(items, f"{group}__chat", group, fam, syc_path.name, "compare_chat", render_chat_with_tokenizer(compare_tok, user_message), user_message=user_message)

    cert_path = data_path("certainty_calibration_items.csv")
    if cert_path.exists():
        with cert_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                item_id = row.get("item_id") or row.get("id") or sha256_text(str(row))[:8]
                question = decode_cell(row.get("question", ""))
                if question:
                    fam = "certainty_" + row.get("family", "unknown")
                    add_prompt(items, f"cert_{item_id}__raw", f"cert_{item_id}", fam, cert_path.name, "raw", question, user_message=question)

    humor_path = data_path("humor_incongruity_pairs.csv")
    if humor_path.exists():
        with humor_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                item_id = row.get("item_id") or row.get("id") or sha256_text(str(row))[:8]
                setup = decode_cell(row.get("setup", ""))
                if setup:
                    user_message = f"Complete this setup with a concise ending. Setup: {setup}\nEnding:"
                    add_prompt(items, f"humor_{item_id}__raw", f"humor_{item_id}", "humor_" + row.get("family", "unknown"), humor_path.name, "raw", user_message, user_message=user_message)

    return items


def prompt_cap(args: Any) -> int:
    cap = int(getattr(args, "max_examples", 0) or 0)
    if cap > 0:
        return cap
    prompt_set = str(getattr(args, "prompt_set", "small") or "small").lower()
    if prompt_set == "small":
        return 32
    if prompt_set == "medium":
        return 80
    return 0


def load_prompt_inventory(args: Any, compare_tok: Any) -> tuple[list[PromptItem], dict[str, Any]]:
    prompt_set = str(getattr(args, "prompt_set", "small") or "small")
    candidate_path = pathlib.Path(prompt_set).expanduser() if prompt_set not in {"small", "medium", "full"} else None

    if candidate_path and candidate_path.exists():
        items = load_custom_prompts(candidate_path, compare_tok)
        source_mode = "custom"
    else:
        items = load_course_prompt_sources(compare_tok) + builtin_prompts(compare_tok)
        source_mode = "course_plus_builtin"

    # Text dedupe can accidentally remove the raw/chat contrast if the tokenizer
    # fallback makes them identical, so dedupe by prompt_id first and by text only
    # inside the same prompt_group and variant.
    seen: set[tuple[str, str, str]] = set()
    deduped: list[PromptItem] = []
    for item in items:
        key = (item.prompt_group, item.variant, item.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    items = deduped

    cap = prompt_cap(args)
    if cap > 0 and len(items) > cap:
        # Stable, mixed-family cap. Keep the selection deterministic while making
        # contiguous rows from one CSV less likely to monopolize the run.
        items = sorted(items, key=lambda r: stable_hash_int(r.prompt_id + str(getattr(args, "seed", 0))))[:cap]
    else:
        items = sorted(items, key=lambda r: (r.family, r.prompt_group, r.variant, r.prompt_id))

    if len(items) < 8:
        raise RuntimeError("Lab 19 needs at least 8 prompts for a meaningful smoke crosscoder.")

    hashes = [sha256_text(item.prompt_id + "\t" + item.text) for item in items]
    info = {
        "source_mode": source_mode,
        "custom_path": str(candidate_path) if candidate_path else "",
        "n_prompts": len(items),
        "counts_by_family": dict(Counter(item.family for item in items)),
        "counts_by_variant": dict(Counter(item.variant for item in items)),
        "counts_by_source": dict(Counter(item.source for item in items)),
        "cap": cap,
        "prompt_inventory_sha256": hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest(),
        "selection_rule": "stable mixed-family cap after prompt_group/variant/text dedupe",
    }
    return items, info


# ---------------------------------------------------------------------------
# Model-pair loading and self-checks
# ---------------------------------------------------------------------------


def comparison_model_spec(ctx: bench.RunContext, primary_bundle: bench.ModelBundle) -> tuple[str, str | None, dict[str, Any]]:
    env_model = os.environ.get("LAB19_COMPARE_MODEL")
    env_revision = os.environ.get("LAB19_COMPARE_MODEL_REVISION")
    if env_model:
        return env_model, env_revision, {"source": "LAB19_COMPARE_MODEL"}

    profile = getattr(bench, "LAB_PROFILES", {}).get(ctx.args.lab, {})
    tier_key = f"compare_model_tier_{ctx.args.tier}"
    if profile.get(tier_key):
        return str(profile[tier_key]), env_revision, {"source": f"registry:{tier_key}"}

    if str(getattr(ctx.args, "tier", "a")) == "a":
        return primary_bundle.anatomy.model_id, primary_bundle.anatomy.revision, {
            "source": "tier_a_identity_fallback",
            "note": "Tier A identity pair proves plumbing, not science.",
        }

    return DEFAULT_COMPARE_MODEL, env_revision, {
        "source": "lab19_default_compare_model",
        "note": "Registry did not provide a compare model; using the course OLMo instruct default.",
    }


def load_comparison_bundle(ctx: bench.RunContext, model_id: str, revision: str | None) -> bench.ModelBundle:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = bench.resolve_device(torch, ctx.args.device)
    dtype = bench.resolve_dtype(torch, ctx.args.dtype, device)
    print(f"[lab19] loading comparison model {model_id!r} (device={device}, dtype={dtype})")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=ctx.args.trust_remote_code,
        local_files_only=ctx.args.local_files_only,
    )
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "revision": revision,
        "trust_remote_code": ctx.args.trust_remote_code,
        "local_files_only": ctx.args.local_files_only,
        "torch_dtype": dtype,
    }
    if getattr(ctx.args, "attn_implementation", "auto") != "auto":
        kwargs["attn_implementation"] = ctx.args.attn_implementation
    if getattr(ctx.args, "low_cpu_mem_usage", False):
        kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    if str(device) != "cpu":
        model.to(device)

    anatomy, blocks, final_norm, lm_head = bench.resolve_anatomy(model, model_id, revision)
    return bench.ModelBundle(
        model=model,
        tokenizer=tokenizer,
        anatomy=anatomy,
        blocks=blocks,
        final_norm=final_norm,
        lm_head=lm_head,
        device=device,
        input_device=bench.infer_input_device(model, device),
        lens_device=bench.first_module_device(final_norm) or device,
        torch_dtype=dtype,
        model_device_map=bench.device_map_summary(model),
    )


def maybe_release_primary_model(ctx: bench.RunContext, bundle: bench.ModelBundle, same_model: bool) -> dict[str, Any]:
    """Best-effort memory relief before loading model B.

    The bench still keeps the Python object, but moving it to CPU can keep a
    24GB GPU run from collapsing. This is skipped by default for identity pairs
    and can be disabled with LAB19_KEEP_PRIMARY_ON_GPU=1.
    """

    info = {"attempted": False, "ok": None, "reason": ""}
    if same_model:
        info["reason"] = "identity_pair"
        return info
    if os.environ.get("LAB19_OFFLOAD_PRIMARY_TO_CPU") != "1":
        info["reason"] = "offload_disabled_by_default; set LAB19_OFFLOAD_PRIMARY_TO_CPU=1 if memory requires it"
        return info
    if os.environ.get("LAB19_KEEP_PRIMARY_ON_GPU") == "1":
        info["reason"] = "LAB19_KEEP_PRIMARY_ON_GPU=1"
        return info
    try:
        import torch

        if str(bundle.device) == "cpu":
            info["reason"] = "primary_already_cpu"
            return info
        info["attempted"] = True
        bundle.model.to("cpu")
        bundle.device = "cpu"
        bundle.input_device = torch.device("cpu")
        bundle.lens_device = torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        info["ok"] = True
        info["reason"] = "moved_primary_to_cpu_before_loading_compare"
    except Exception as exc:  # pragma: no cover, architecture/device-map dependent
        info["ok"] = False
        info["reason"] = f"primary_offload_failed: {type(exc).__name__}: {exc}"
    return info


def select_depth(bundle: bench.ModelBundle, role: str) -> int:
    env = os.environ.get(f"LAB19_DEPTH_{role.upper()}") or os.environ.get("LAB19_DEPTH")
    if env:
        depth = int(env)
    else:
        depth = max(1, int(round(bundle.anatomy.n_layers * 0.65)))
    return max(1, min(bundle.anatomy.n_layers, depth))


def run_prefixed_hook_parity_check(ctx: bench.RunContext, bundle: bench.ModelBundle, prompt: str, prefix: str) -> dict[str, Any]:
    import torch

    block_outputs: dict[int, Any] = {}

    def make_hook(layer: int):
        def hook(_module: Any, _inp: Any, out: Any) -> None:
            value = out[0] if isinstance(out, tuple) else out
            block_outputs[layer] = value.detach().to(device="cpu", dtype=torch.float32)
        return hook

    handles = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(bundle.blocks)]
    try:
        capture = bench.run_with_residual_cache(bundle, prompt, add_special_tokens=False)
    finally:
        for handle in handles:
            handle.remove()

    rows: list[dict[str, Any]] = []
    max_diff = 0.0
    max_mean = 0.0
    missing: list[int] = []
    compared = 0
    for layer in range(bundle.anatomy.n_layers):
        if layer not in block_outputs:
            missing.append(layer)
            continue
        expected = capture.streams[layer + 1]
        got = block_outputs[layer][0]
        diff = (got - expected).abs()
        layer_max = float(diff.max())
        layer_mean = float(diff.mean())
        max_diff = max(max_diff, layer_max)
        max_mean = max(max_mean, layer_mean)
        compared += 1
        rows.append({
            "model_prefix": prefix,
            "layer": layer,
            "stream_depth": layer + 1,
            "max_abs_diff": layer_max,
            "mean_abs_diff": layer_mean,
            "ok_at_tolerance": layer_max <= ctx.args.hook_tolerance,
            "note": "block layer output is streams[layer + 1]",
        })

    by_path = ctx.path("diagnostics", f"{prefix}_hook_parity_by_layer.csv")
    bench.write_csv_with_context(ctx, by_path, rows)
    ctx.register_artifact(by_path, "diagnostic", f"{prefix} layer-level hook parity for rendered Lab 19 prompt.")

    ok = (not missing) and compared == bundle.anatomy.n_layers and max_diff <= ctx.args.hook_tolerance
    payload = {
        "model_prefix": prefix,
        "model_id": bundle.anatomy.model_id,
        "blocks_compared": compared,
        "n_layers": bundle.anatomy.n_layers,
        "missing_layers": missing,
        "max_abs_diff": max_diff,
        "max_mean_abs_diff": max_mean,
        "tolerance": ctx.args.hook_tolerance,
        "ok": bool(ok),
        "allow_hook_mismatch": bool(ctx.args.allow_hook_mismatch),
        "prompt_sha256": sha256_text(prompt),
        "explanation": "Forward hooks on block outputs are compared with streams[layer + 1] on the exact rendered prompt.",
    }
    path = ctx.path("diagnostics", f"{prefix}_hook_parity.json")
    bench.write_json(path, payload)
    ctx.register_artifact(path, "diagnostic", f"{prefix} hook parity check for Lab 19.")
    print(f"[lab19] {prefix} hook parity: {'OK' if ok else 'MISMATCH'} max |diff|={max_diff:g}")
    if not ok and not ctx.args.allow_hook_mismatch:
        raise RuntimeError(f"{prefix} hook parity failed. See diagnostics/{prefix}_hook_parity.json")
    return payload


def run_prefixed_lens_self_check(ctx: bench.RunContext, bundle: bench.ModelBundle, prompt: str, prefix: str) -> dict[str, Any]:
    import torch

    capture = bench.run_with_residual_cache(bundle, prompt, add_special_tokens=False)
    lens_logits = bench.logit_lens_all_depths(bundle, capture.streams[:, -1, :])
    lens_final = lens_logits[-1]
    real_final = capture.final_logits_last
    diff = (lens_final - real_final).abs()
    lens_top = int(torch.argmax(lens_final).item())
    real_top = int(torch.argmax(real_final).item())
    payload = {
        "model_prefix": prefix,
        "model_id": bundle.anatomy.model_id,
        "prompt_sha256": sha256_text(prompt),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "top1_agrees": bool(lens_top == real_top),
        "lens_top1_token_id": lens_top,
        "real_top1_token_id": real_top,
        "lens_top1_piece": bundle.tokenizer.decode([lens_top]),
        "real_top1_piece": bundle.tokenizer.decode([real_top]),
        "ok": bool(lens_top == real_top),
        "explanation": "The lens at final stream depth L should reproduce the model's actual final logits top token.",
    }
    path = ctx.path("diagnostics", f"{prefix}_logit_lens_self_check.json")
    bench.write_json(path, payload)
    ctx.register_artifact(path, "diagnostic", f"{prefix} final-depth logit-lens self-check for Lab 19.")
    if not payload["ok"]:
        raise RuntimeError(f"{prefix} final-depth lens self-check failed. See {path}")
    return payload


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------


def make_split(prompts: Sequence[PromptItem], seed: int) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(prompts):
        groups[item.prompt_group].append(i)
    ordered_groups = sorted(groups, key=lambda g: stable_hash_int(f"{seed}:{g}"))
    n_eval_groups = max(1, int(round(len(ordered_groups) * EVAL_FRACTION))) if len(ordered_groups) > 3 else 1
    eval_groups = set(ordered_groups[:n_eval_groups])
    eval_idx = sorted(i for g in eval_groups for i in groups[g])
    train_idx = sorted(i for g in groups if g not in eval_groups for i in groups[g])
    if len(train_idx) < 4 and len(prompts) >= 8:
        eval_idx = sorted(i for i in range(len(prompts)) if i % 4 == 0)
        train_idx = sorted(i for i in range(len(prompts)) if i not in set(eval_idx))
    return {"train": train_idx, "eval": eval_idx}


def split_rows(prompts: Sequence[PromptItem], split: Mapping[str, Sequence[int]]) -> list[dict[str, Any]]:
    idx_to_split = {i: name for name, idxs in split.items() for i in idxs}
    rows = []
    for i, item in enumerate(prompts):
        rows.append({
            "row_index": i,
            "split": idx_to_split.get(i, "unassigned"),
            "prompt_id": item.prompt_id,
            "prompt_group": item.prompt_group,
            "family": item.family,
            "variant": item.variant,
            "source": item.source,
            "prompt_hash": sha256_text(item.text),
        })
    return rows


def collect_model_activations(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    prompts: Sequence[PromptItem],
    depth: int,
    prefix: str,
) -> tuple[Any, list[dict[str, Any]]]:
    import torch

    xs = []
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(prompts):
        capture = bench.run_with_residual_cache(bundle, item.text, add_special_tokens=False)
        stream = capture.streams[depth, -1, :].detach().to(device="cpu", dtype=torch.float32)
        xs.append(stream)
        ids = token_ids(bundle.tokenizer, item.text)
        rows.append({
            "model_prefix": prefix,
            "row_index": i,
            "prompt_id": item.prompt_id,
            "prompt_group": item.prompt_group,
            "family": item.family,
            "source": item.source,
            "variant": item.variant,
            "stream_depth": depth,
            "n_tokens": len(ids),
            "last_token_id": ids[-1] if ids else "",
            "last_token_piece": bundle.tokenizer.decode([ids[-1]]) if ids else "",
            "residual_norm": float(stream.norm()),
            "residual_mean": float(stream.mean()),
            "residual_std": float(stream.std(unbiased=False)),
            "text_sha256": sha256_text(item.text),
            "text_excerpt": item.text[-220:].replace("\n", "\\n"),
        })
        if (i + 1) % 20 == 0:
            print(f"[lab19] {prefix} captured {i + 1}/{len(prompts)} prompts at stream depth {depth}")
    return torch.stack(xs, dim=0), rows


def activation_norm_control_rows(rows_a: Sequence[Mapping[str, Any]], rows_b: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_row_b = {int(r["row_index"]): r for r in rows_b}
    for ra in rows_a:
        rb = by_row_b.get(int(ra["row_index"]))
        if rb is None:
            continue
        key = (str(ra["family"]), str(ra["variant"]))
        by_key[key]["norm_a"].append(float(ra["residual_norm"]))
        by_key[key]["norm_b"].append(float(rb["residual_norm"]))
        by_key[key]["n_tokens_a"].append(float(ra["n_tokens"]))
        by_key[key]["n_tokens_b"].append(float(rb["n_tokens"]))
    out = []
    for (family, variant), vals in sorted(by_key.items()):
        ma = safe_fmean(vals["norm_a"])
        mb = safe_fmean(vals["norm_b"])
        out.append({
            "family": family,
            "variant": variant,
            "n_prompts": len(vals["norm_a"]),
            "mean_norm_model_a": rounded(ma),
            "mean_norm_model_b": rounded(mb),
            "mean_norm_ratio_b_over_a": rounded(mb / ma if ma and math.isfinite(ma) else float("nan")),
            "mean_tokens_model_a": rounded(safe_fmean(vals["n_tokens_a"])),
            "mean_tokens_model_b": rounded(safe_fmean(vals["n_tokens_b"])),
        })
    return out


def collect_pair_activations(
    ctx: bench.RunContext,
    bundle_a: bench.ModelBundle,
    bundle_b: bench.ModelBundle,
    prompts: Sequence[PromptItem],
    depth_a: int,
    depth_b: int,
    split: Mapping[str, Sequence[int]],
    *,
    same_model: bool,
) -> PairActivations:
    x_a, rows_a = collect_model_activations(ctx, bundle_a, prompts, depth_a, "model_a")
    if same_model and depth_a == depth_b:
        x_b = x_a.clone()
        rows_b = [{**row, "model_prefix": "model_b", "note": "reused identity-pair activation from model_a"} for row in rows_a]
    else:
        x_b, rows_b = collect_model_activations(ctx, bundle_b, prompts, depth_b, "model_b")
    joined: list[dict[str, Any]] = []
    by_row_b = {int(r["row_index"]): r for r in rows_b}
    split_lookup = {i: name for name, idxs in split.items() for i in idxs}
    for ra in rows_a:
        i = int(ra["row_index"])
        rb = by_row_b[i]
        norm_a = float(ra["residual_norm"])
        norm_b = float(rb["residual_norm"])
        joined.append({
            "row_index": i,
            "split": split_lookup.get(i, "unassigned"),
            "prompt_id": ra["prompt_id"],
            "prompt_group": ra["prompt_group"],
            "family": ra["family"],
            "source": ra["source"],
            "variant": ra["variant"],
            "n_tokens_model_a": ra["n_tokens"],
            "n_tokens_model_b": rb["n_tokens"],
            "last_token_piece_model_a": ra["last_token_piece"],
            "last_token_piece_model_b": rb["last_token_piece"],
            "depth_a": depth_a,
            "depth_b": depth_b,
            "norm_model_a": rounded(norm_a),
            "norm_model_b": rounded(norm_b),
            "norm_ratio_b_over_a": rounded(norm_b / norm_a if norm_a > 1e-9 else float("nan")),
            "text_sha256": ra["text_sha256"],
            "text_excerpt": ra["text_excerpt"],
        })
    return PairActivations(joined, x_a, x_b, depth_a, depth_b, dict(split))


def pair_activations_from_parts(
    x_a: Any,
    rows_a: Sequence[Mapping[str, Any]],
    x_b: Any,
    rows_b: Sequence[Mapping[str, Any]],
    depth_a: int,
    depth_b: int,
    split: Mapping[str, Sequence[int]],
) -> PairActivations:
    """Join precomputed model-A and model-B activations without re-forwarding.

    This lets the lab capture model A, move it off GPU, then load model B.
    A full OLMo base+instruct pair rarely fits on one 24GB card at once.
    """

    joined: list[dict[str, Any]] = []
    by_row_b = {int(r["row_index"]): r for r in rows_b}
    split_lookup = {i: name for name, idxs in split.items() for i in idxs}
    for ra in rows_a:
        i = int(ra["row_index"])
        rb = by_row_b[i]
        norm_a = float(ra["residual_norm"])
        norm_b = float(rb["residual_norm"])
        joined.append({
            "row_index": i,
            "split": split_lookup.get(i, "unassigned"),
            "prompt_id": ra["prompt_id"],
            "prompt_group": ra["prompt_group"],
            "family": ra["family"],
            "source": ra["source"],
            "variant": ra["variant"],
            "n_tokens_model_a": ra["n_tokens"],
            "n_tokens_model_b": rb["n_tokens"],
            "last_token_piece_model_a": ra["last_token_piece"],
            "last_token_piece_model_b": rb["last_token_piece"],
            "depth_a": depth_a,
            "depth_b": depth_b,
            "norm_model_a": rounded(norm_a),
            "norm_model_b": rounded(norm_b),
            "norm_ratio_b_over_a": rounded(norm_b / norm_a if norm_a > 1e-9 else float("nan")),
            "text_sha256": ra["text_sha256"],
            "text_excerpt": ra["text_excerpt"],
        })
    return PairActivations(joined, x_a, x_b, depth_a, depth_b, dict(split))


# ---------------------------------------------------------------------------
# Crosscoder
# ---------------------------------------------------------------------------


class PairedCrosscoder:
    """A tiny tied-index paired sparse dictionary.

    `encode_pair` creates one shared feature vector from the concatenated pair.
    The same feature index reconstructs both model activations through separate
    decoders. Side-only encoders are used for feature taxonomy, not as the main
    reconstruction path.
    """

    def __init__(self, d_a: int, d_b: int, n_features: int, torch: Any, seed: int):
        g = torch.Generator(device="cpu").manual_seed(seed)
        scale_a = 1.0 / math.sqrt(max(1, d_a))
        scale_b = 1.0 / math.sqrt(max(1, d_b))
        self.W_a = torch.randn(d_a, n_features, generator=g) * scale_a
        self.W_b = torch.randn(d_b, n_features, generator=g) * scale_b
        self.b = torch.zeros(n_features)
        self.D_a = torch.randn(n_features, d_a, generator=g) * scale_a
        self.D_b = torch.randn(n_features, d_b, generator=g) * scale_b
        for p in self.parameters():
            p.requires_grad_(True)

    def parameters(self) -> list[Any]:
        return [self.W_a, self.W_b, self.b, self.D_a, self.D_b]

    def to(self, device: Any) -> "PairedCrosscoder":
        for name in ("W_a", "W_b", "b", "D_a", "D_b"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def encode_pair(self, xa: Any, xb: Any) -> Any:
        import torch
        return torch.relu(xa @ self.W_a + xb @ self.W_b + self.b)

    def encode_a(self, xa: Any) -> Any:
        import torch
        return torch.relu(xa @ self.W_a + self.b)

    def encode_b(self, xb: Any) -> Any:
        import torch
        return torch.relu(xb @ self.W_b + self.b)

    def decode_a(self, z: Any) -> Any:
        return z @ self.D_a

    def decode_b(self, z: Any) -> Any:
        return z @ self.D_b

    def detach_cpu(self) -> "PairedCrosscoder":
        import torch
        clone = PairedCrosscoder(1, 1, 1, torch, 0)
        clone.W_a = self.W_a.detach().cpu()
        clone.W_b = self.W_b.detach().cpu()
        clone.b = self.b.detach().cpu()
        clone.D_a = self.D_a.detach().cpu()
        clone.D_b = self.D_b.detach().cpu()
        return clone


def fvu(x: Any, recon: Any) -> float:
    import torch
    resid = torch.mean((x - recon) ** 2)
    denom = torch.var(x, unbiased=False).clamp_min(1e-9)
    return float((resid / denom).detach().cpu())


def normalized_train_eval(acts: PairActivations) -> dict[str, Any]:
    import torch

    train_idx = acts.split.get("train") or list(range(acts.x_a.shape[0]))
    train = torch.tensor(train_idx, dtype=torch.long)
    mean_a = acts.x_a[train].mean(dim=0, keepdim=True)
    mean_b = acts.x_b[train].mean(dim=0, keepdim=True)
    std_a = acts.x_a[train].std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
    std_b = acts.x_b[train].std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
    return {
        "xa": (acts.x_a - mean_a) / std_a,
        "xb": (acts.x_b - mean_b) / std_b,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "std_a": std_a,
        "std_b": std_b,
    }


def crosscoder_steps(args: Any) -> int:
    forced = os.environ.get("LAB19_TRAIN_STEPS")
    if forced:
        return int(forced)
    return TRAIN_STEPS_TIER_A if getattr(args, "tier", "a") == "a" else TRAIN_STEPS_TIER_B


def crosscoder_features(args: Any, n_train: int) -> int:
    forced = os.environ.get("LAB19_FEATURES")
    n = int(forced) if forced else CROSSCODER_FEATURES
    if getattr(args, "tier", "a") == "a":
        n = min(n, 80)
    # Tiny dictionaries are boring, huge dictionaries in a tiny smoke run create
    # arbitrary exclusive atoms. Cap by data size but keep enough width to show
    # asymmetry and dead features.
    return max(16, min(n, max(16, n_train * 6)))


def train_crosscoder(ctx: bench.RunContext, acts: PairActivations, seed: int) -> tuple[PairedCrosscoder, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    import torch

    norm = normalized_train_eval(acts)
    xa = norm["xa"]
    xb = norm["xb"]
    train_idx = acts.split.get("train") or list(range(xa.shape[0]))
    eval_idx = acts.split.get("eval") or []
    n_features = crosscoder_features(ctx.args, len(train_idx))
    model = PairedCrosscoder(xa.shape[1], xb.shape[1], n_features, torch, seed)
    device = torch.device("cpu")
    model.to(device)
    xa = xa.to(device)
    xb = xb.to(device)
    train_tensor = torch.tensor(train_idx, dtype=torch.long, device=device)
    eval_tensor = torch.tensor(eval_idx, dtype=torch.long, device=device) if eval_idx else train_tensor
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps = crosscoder_steps(ctx.args)
    curve: list[dict[str, Any]] = []

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        ta = xa[train_tensor]
        tb = xb[train_tensor]
        z_pair = model.encode_pair(ta, tb)
        z_a = model.encode_a(ta)
        z_b = model.encode_b(tb)
        ra = model.decode_a(z_pair)
        rb = model.decode_b(z_pair)
        sa = model.decode_a(z_a)
        sb = model.decode_b(z_b)
        pair_recon = torch.mean((ra - ta) ** 2) + torch.mean((rb - tb) ** 2)
        side_recon = torch.mean((sa - ta) ** 2) + torch.mean((sb - tb) ** 2)
        l1 = (z_pair.abs().mean() + 0.5 * z_a.abs().mean() + 0.5 * z_b.abs().mean())
        loss = pair_recon + SIDE_RECON_WEIGHT * side_recon + L1_WEIGHT * l1
        loss.backward()
        opt.step()
        if step == 0 or step == steps - 1 or (step + 1) % max(20, steps // 8) == 0:
            with torch.no_grad():
                ez = model.encode_pair(xa[eval_tensor], xb[eval_tensor])
                curve.append({
                    "step": step + 1,
                    "loss": rounded(float(loss.detach().cpu())),
                    "train_fvu_model_a": rounded(fvu(ta, model.decode_a(z_pair))),
                    "train_fvu_model_b": rounded(fvu(tb, model.decode_b(z_pair))),
                    "eval_fvu_model_a": rounded(fvu(xa[eval_tensor], model.decode_a(ez))),
                    "eval_fvu_model_b": rounded(fvu(xb[eval_tensor], model.decode_b(ez))),
                    "mean_pair_sparsity": rounded(float((z_pair > 1e-6).float().mean().detach().cpu())),
                    "mean_side_sparsity_a": rounded(float((z_a > 1e-6).float().mean().detach().cpu())),
                    "mean_side_sparsity_b": rounded(float((z_b > 1e-6).float().mean().detach().cpu())),
                })

    with torch.no_grad():
        z_pair = model.encode_pair(xa, xb)
        z_a = model.encode_a(xa)
        z_b = model.encode_b(xb)
        recon_a = model.decode_a(z_pair)
        recon_b = model.decode_b(z_pair)
        train_z = model.encode_pair(xa[train_tensor], xb[train_tensor])
        eval_z = model.encode_pair(xa[eval_tensor], xb[eval_tensor])
        train_metrics = {
            "n_features": n_features,
            "train_steps": steps,
            "l1_weight": L1_WEIGHT,
            "side_recon_weight": SIDE_RECON_WEIGHT,
            "train_fvu_model_a": rounded(fvu(xa[train_tensor], model.decode_a(train_z))),
            "train_fvu_model_b": rounded(fvu(xb[train_tensor], model.decode_b(train_z))),
            "eval_fvu_model_a": rounded(fvu(xa[eval_tensor], model.decode_a(eval_z))) if eval_idx else None,
            "eval_fvu_model_b": rounded(fvu(xb[eval_tensor], model.decode_b(eval_z))) if eval_idx else None,
            "fvu_model_a": rounded(fvu(xa, recon_a)),
            "fvu_model_b": rounded(fvu(xb, recon_b)),
            "mean_pair_feature_density": rounded(float((z_pair > 1e-6).float().mean().detach().cpu())),
            "mean_side_feature_density_a": rounded(float((z_a > 1e-6).float().mean().detach().cpu())),
            "mean_side_feature_density_b": rounded(float((z_b > 1e-6).float().mean().detach().cpu())),
        }
        stats = {
            **norm,
            "z_pair": z_pair.detach().cpu(),
            "z_a": z_a.detach().cpu(),
            "z_b": z_b.detach().cpu(),
        }
    return model.detach_cpu(), train_metrics, {k: (v.detach().cpu() if hasattr(v, "detach") else v) for k, v in stats.items()}, curve


def classify_feature(a_mean: float, b_mean: float, corr: float, decoder_b_share: float, pair_mean: float) -> str:
    total = a_mean + b_mean
    if pair_mean < 1e-5 and total < 1e-5:
        return "dead"
    activation_b_share = b_mean / total if total > 1e-9 else 0.5
    combined_b = 0.55 * activation_b_share + 0.45 * decoder_b_share
    if 0.38 <= activation_b_share <= 0.62 and 0.35 <= decoder_b_share <= 0.65 and (math.isnan(corr) or corr >= 0.25):
        return "shared"
    if combined_b >= 0.72 and (activation_b_share >= 0.62 or decoder_b_share >= 0.78):
        return "model_b_only"
    if combined_b <= 0.28 and (activation_b_share <= 0.38 or decoder_b_share <= 0.22):
        return "model_a_only"
    return "asymmetric"


def top_concentration(scores: Sequence[float], prompts: Sequence[PromptItem], attr: str, top_n: int = 8) -> tuple[str, float]:
    if not scores:
        return "", float("nan")
    top = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[: min(top_n, len(scores))]
    vals = [getattr(prompts[i], attr) for i in top]
    counts = Counter(vals)
    label, count = counts.most_common(1)[0]
    return label, count / max(1, len(vals))


def feature_taxonomy_rows(
    model: PairedCrosscoder,
    stats: Mapping[str, Any],
    prompts: Sequence[PromptItem],
    split: Mapping[str, Sequence[int]],
    role_a: str,
    role_b: str,
) -> list[dict[str, Any]]:
    import torch

    z_a = stats["z_a"]
    z_b = stats["z_b"]
    z_pair = stats["z_pair"]
    std_a = stats["std_a"].reshape(-1)
    std_b = stats["std_b"].reshape(-1)
    dec_a_resid = model.D_a * std_a.reshape(1, -1)
    dec_b_resid = model.D_b * std_b.reshape(1, -1)
    dec_norm_a = torch.linalg.vector_norm(dec_a_resid, dim=1)
    dec_norm_b = torch.linalg.vector_norm(dec_b_resid, dim=1)
    rows: list[dict[str, Any]] = []
    train = split.get("train", [])
    eval_idx = split.get("eval", [])
    for fid in range(z_a.shape[1]):
        a_vals = [float(x) for x in z_a[:, fid].tolist()]
        b_vals = [float(x) for x in z_b[:, fid].tolist()]
        pair_vals = [float(x) for x in z_pair[:, fid].tolist()]
        a_mean = safe_fmean(a_vals, 0.0)
        b_mean = safe_fmean(b_vals, 0.0)
        pair_mean = safe_fmean(pair_vals, 0.0)
        corr = pearson(a_vals, b_vals)
        total = a_mean + b_mean
        activation_b_share = b_mean / total if total > 1e-9 else float("nan")
        decoder_b_share = float(dec_norm_b[fid] / (dec_norm_a[fid] + dec_norm_b[fid] + 1e-9))
        generic = classify_feature(a_mean, b_mean, corr, decoder_b_share, pair_mean)
        role_tax = generic_to_role_taxonomy(generic, role_a, role_b)
        fam_b, fam_conc_b = top_concentration(b_vals, prompts, "family")
        var_b, var_conc_b = top_concentration(b_vals, prompts, "variant")
        train_activity = safe_fmean([pair_vals[i] for i in train], 0.0) if train else float("nan")
        eval_activity = safe_fmean([pair_vals[i] for i in eval_idx], 0.0) if eval_idx else float("nan")
        rows.append({
            "feature_id": fid,
            "taxonomy": generic,
            "role_taxonomy": role_tax,
            "model_a_role": role_a,
            "model_b_role": role_b,
            "activation_mean_model_a": rounded(a_mean),
            "activation_mean_model_b": rounded(b_mean),
            "pair_activation_mean": rounded(pair_mean),
            "model_b_activation_share": rounded(activation_b_share),
            "activation_correlation_a_b": rounded(corr),
            "decoder_norm_model_a_residual_units": rounded(float(dec_norm_a[fid])),
            "decoder_norm_model_b_residual_units": rounded(float(dec_norm_b[fid])),
            "decoder_norm_model_b_share": rounded(decoder_b_share),
            "model_b_specificity_score": rounded(0.55 * (activation_b_share if math.isfinite(activation_b_share) else 0.5) + 0.45 * decoder_b_share),
            "top_model_b_family": fam_b,
            "top_model_b_family_concentration": rounded(fam_conc_b),
            "top_model_b_variant": var_b,
            "top_model_b_variant_concentration": rounded(var_conc_b),
            "train_pair_activity": rounded(train_activity),
            "eval_pair_activity": rounded(eval_activity),
            "eval_over_train_activity": rounded(eval_activity / train_activity if train_activity > 1e-9 and math.isfinite(eval_activity) else float("nan")),
            "audit_flag_template_concentrated": bool(var_conc_b >= 0.75 and var_b == "compare_chat"),
            "audit_flag_family_concentrated": bool(fam_conc_b >= 0.75),
        })
    rows.sort(key=lambda r: (str(r["taxonomy"]), -float(r["model_b_specificity_score"] or 0.0), int(r["feature_id"])))
    return rows


def feature_stability_rows(taxonomy: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in taxonomy:
        ratio = row.get("eval_over_train_activity")
        try:
            low_eval_activity = float(ratio) < 0.25
        except Exception:
            low_eval_activity = True
        rows.append({
            "feature_id": row["feature_id"],
            "taxonomy": row["taxonomy"],
            "role_taxonomy": row["role_taxonomy"],
            "train_pair_activity": row["train_pair_activity"],
            "eval_pair_activity": row["eval_pair_activity"],
            "eval_over_train_activity": row["eval_over_train_activity"],
            "top_model_b_family": row["top_model_b_family"],
            "top_model_b_family_concentration": row["top_model_b_family_concentration"],
            "top_model_b_variant": row["top_model_b_variant"],
            "top_model_b_variant_concentration": row["top_model_b_variant_concentration"],
            "stability_warning": "train_only_feature" if low_eval_activity else "",
        })
    return rows


def gallery_rows(taxonomy: Sequence[Mapping[str, Any]], stats: Mapping[str, Any], prompts: Sequence[PromptItem]) -> list[dict[str, Any]]:
    z_a = stats["z_a"]
    z_b = stats["z_b"]
    candidates = sorted(
        [r for r in taxonomy if r["taxonomy"] in {"model_b_only", "model_a_only", "shared", "asymmetric"}],
        key=lambda r: (0 if r["taxonomy"] == "model_b_only" else 1, -float(r.get("model_b_specificity_score") or 0.0)),
    )[:GALLERY_FEATURES]
    rows: list[dict[str, Any]] = []
    for row in candidates:
        fid = int(row["feature_id"])
        if row["taxonomy"] == "model_a_only":
            scores = z_a[:, fid]
        elif row["taxonomy"] == "model_b_only":
            scores = z_b[:, fid]
        else:
            scores = z_a[:, fid] + z_b[:, fid]
        top = sorted(range(len(prompts)), key=lambda i: float(scores[i]), reverse=True)[:GALLERY_CONTEXTS]
        for rank, idx in enumerate(top, start=1):
            prompt = prompts[idx]
            rows.append({
                "feature_id": fid,
                "taxonomy": row["taxonomy"],
                "role_taxonomy": row["role_taxonomy"],
                "rank": rank,
                "score_model_a": rounded(float(z_a[idx, fid])),
                "score_model_b": rounded(float(z_b[idx, fid])),
                "family": prompt.family,
                "variant": prompt.variant,
                "source": prompt.source,
                "prompt_id": prompt.prompt_id,
                "prompt_group": prompt.prompt_group,
                "text_excerpt": prompt.text[:260].replace("\n", "\\n"),
                "candidate_label_axes": "template|refusal|politeness|hedging|disclosure|self_situation|default_voice|topic|other",
                "student_proposed_label": "",
                "student_label_status": "unlabeled",
                "student_counterexample": "",
            })
    return rows


def template_control_summary(taxonomy: Sequence[Mapping[str, Any]], stats: Mapping[str, Any], prompts: Sequence[PromptItem]) -> list[dict[str, Any]]:
    z_b = stats["z_b"]
    by_group: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, item in enumerate(prompts):
        by_group[item.prompt_group][item.variant].append(i)
    paired = [v for v in by_group.values() if v.get("raw") and v.get("compare_chat")]
    rows: list[dict[str, Any]] = []
    by_fid = {int(r["feature_id"]): r for r in taxonomy}
    for fid in range(z_b.shape[1]):
        diffs = []
        for group in paired:
            raw_mean = float(z_b[group["raw"], fid].mean())
            chat_mean = float(z_b[group["compare_chat"], fid].mean())
            diffs.append(chat_mean - raw_mean)
        mean_gap = safe_fmean(diffs)
        rows.append({
            "feature_id": fid,
            "taxonomy": by_fid.get(fid, {}).get("taxonomy", ""),
            "role_taxonomy": by_fid.get(fid, {}).get("role_taxonomy", ""),
            "n_raw_chat_pairs": len(diffs),
            "mean_compare_chat_minus_raw_activation_model_b": rounded(mean_gap),
            "abs_template_gap": rounded(abs(mean_gap) if math.isfinite(mean_gap) else float("nan")),
            "template_dominated_warning": bool(math.isfinite(mean_gap) and abs(mean_gap) > 0.5),
            "note": "Positive means the feature fires more on the chat-rendered variant than on the raw text variant.",
        })
    rows.sort(key=lambda r: -float(r.get("abs_template_gap") or 0.0))
    return rows


def random_feature_baseline_rows(acts: PairActivations, seed: int, n_draws: int = RANDOM_BASELINE_DRAWS) -> list[dict[str, Any]]:
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed + 19019)
    xa = acts.x_a
    xb = acts.x_b
    rows: list[dict[str, Any]] = []
    for draw in range(n_draws):
        va = torch.randn(xa.shape[1], generator=g)
        vb = torch.randn(xb.shape[1], generator=g)
        va = va / va.norm().clamp_min(1e-9)
        vb = vb / vb.norm().clamp_min(1e-9)
        za = torch.relu(xa @ va)
        zb = torch.relu(xb @ vb)
        a_mean = float(za.mean())
        b_mean = float(zb.mean())
        total = a_mean + b_mean
        share_b = b_mean / total if total > 1e-9 else float("nan")
        corr = pearson([float(x) for x in za.tolist()], [float(x) for x in zb.tolist()])
        rows.append({
            "draw": draw,
            "activation_mean_model_a": rounded(a_mean),
            "activation_mean_model_b": rounded(b_mean),
            "model_b_activation_share": rounded(share_b),
            "activation_correlation_a_b": rounded(corr),
            "would_look_model_b_specific": bool(math.isfinite(share_b) and share_b >= 0.72),
            "would_look_model_a_specific": bool(math.isfinite(share_b) and share_b <= 0.28),
        })
    return rows


def voice_marker_rows(prompts: Sequence[PromptItem]) -> list[dict[str, Any]]:
    rows = []
    for family in sorted({p.family for p in prompts}):
        for variant in sorted({p.variant for p in prompts if p.family == family}):
            sub = [p for p in prompts if p.family == family and p.variant == variant]
            rows.append({
                "family": family,
                "variant": variant,
                "n_prompts": len(sub),
                "prompt_text_default_voice_marker_rate": rounded(safe_fmean(marker_any(p.text, VOICE_MARKERS) for p in sub)),
                "prompt_text_politeness_marker_rate": rounded(safe_fmean(marker_any(p.text, POLITENESS_MARKERS) for p in sub)),
                "prompt_text_hedging_marker_rate": rounded(safe_fmean(marker_any(p.text, HEDGING_MARKERS) for p in sub)),
                "prompt_text_refusal_marker_rate": rounded(safe_fmean(marker_any(p.text, REFUSAL_MARKERS) for p in sub)),
                "note": "Prompt-text marker control only. It does not score generated behavior.",
            })
    return rows


# ---------------------------------------------------------------------------
# Direction bridge
# ---------------------------------------------------------------------------


def recursive_tensors(obj: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(obj, torch.Tensor):
        yield prefix or "tensor", obj
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            yield from recursive_tensors(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            yield from recursive_tensors(value, f"{prefix}[{i}]")


def load_bridge_directions(path: pathlib.Path, d_model: int) -> list[tuple[str, Any]]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu")
    except TypeError:  # older torch without weights_only keyword compatibility issues
        payload = torch.load(path, map_location="cpu")
    dirs = []
    for name, tensor in recursive_tensors(payload):
        t = tensor.detach().float().cpu().reshape(-1)
        if t.numel() == d_model and float(t.norm()) > 1e-9:
            dirs.append((name, t / t.norm().clamp_min(1e-9)))
    return dirs


def direction_bridge_rows(
    taxonomy: Sequence[Mapping[str, Any]],
    model: PairedCrosscoder,
    stats: Mapping[str, Any],
    bundle_b: bench.ModelBundle,
) -> list[dict[str, Any]]:
    import torch

    std_b = stats["std_b"].reshape(-1)
    dec_b = model.D_b * std_b.reshape(1, -1)
    dec_b = dec_b / torch.linalg.vector_norm(dec_b, dim=1, keepdim=True).clamp_min(1e-9)
    top_features = sorted(
        [r for r in taxonomy if r["taxonomy"] in {"model_b_only", "asymmetric", "shared"}],
        key=lambda r: -float(r.get("model_b_specificity_score") or 0.0),
    )[: min(24, len(taxonomy))]
    bridge_path_env = os.environ.get("LAB19_BRIDGE_STATE")
    rows: list[dict[str, Any]] = []
    if bridge_path_env:
        bridge_path = pathlib.Path(bridge_path_env).expanduser()
        if bridge_path.exists():
            dirs = load_bridge_directions(bridge_path, bundle_b.anatomy.d_model)
            for direction_name, direction in dirs:
                for feat in top_features:
                    fid = int(feat["feature_id"])
                    cos = float(torch.dot(dec_b[fid], direction))
                    rows.append({
                        "bridge_state_path": str(bridge_path),
                        "direction_name": direction_name,
                        "status": "computed",
                        "feature_id": fid,
                        "taxonomy": feat["taxonomy"],
                        "role_taxonomy": feat["role_taxonomy"],
                        "cosine_with_model_b_decoder": rounded(cos),
                        "abs_cosine": rounded(abs(cos)),
                        "feature_decoder_space": "model_b_residual_units_at_selected_depth",
                    })
            rows.sort(key=lambda r: -float(r.get("abs_cosine") or 0.0))
            return rows
        rows.append({
            "bridge_state_path": str(bridge_path),
            "direction_name": "",
            "status": "configured_path_missing",
            "feature_id": "",
            "taxonomy": "",
            "role_taxonomy": "",
            "cosine_with_model_b_decoder": "",
            "feature_decoder_space": "model_b_residual_units_at_selected_depth",
            "note": "Set LAB19_BRIDGE_STATE to an existing prior-lab state .pt file.",
        })
        return rows

    for bridge in ("lab07_refusal_or_sentiment", "lab14_certainty", "lab16_user_belief_or_agreement", "lab17_persona_voice_register", "lab18_humor"):
        rows.append({
            "bridge_state_path": "",
            "direction_name": bridge,
            "status": "not_configured",
            "feature_id": "",
            "taxonomy": "",
            "role_taxonomy": "",
            "cosine_with_model_b_decoder": "",
            "feature_decoder_space": "model_b_residual_units_at_selected_depth",
            "note": "Set LAB19_BRIDGE_STATE=/path/to/state.pt to compute feature-direction cosines.",
        })
    return rows


# ---------------------------------------------------------------------------
# Optional causal validation
# ---------------------------------------------------------------------------


def benign_causal_prompts(bundle: bench.ModelBundle) -> list[tuple[str, str]]:
    user_messages = [
        ("organize_notes", "How should I organize project notes after a meeting?"),
        ("uncertain_claim", "How should I answer when I am not sure whether a factual claim is correct?"),
        ("polite_correction", "Write a concise, polite correction to a teammate who has the wrong date."),
        ("study_plan", "Give me a three-step plan for studying an unfamiliar technical paper."),
        ("boundary_private", "A classmate asks for private student records. How should I respond?"),
    ]
    return [(pid, render_for_bundle(bundle, msg)) for pid, msg in user_messages]


def score_generation(text: str) -> dict[str, Any]:
    words = text.split()
    return {
        "default_voice_marker": marker_any(text, VOICE_MARKERS),
        "politeness_marker": marker_any(text, POLITENESS_MARKERS),
        "hedging_marker": marker_any(text, HEDGING_MARKERS),
        "refusal_marker": marker_any(text, REFUSAL_MARKERS),
        "self_situation_marker": marker_any(text, SELF_SITUATION_MARKERS),
        "disclosure_marker": marker_any(text, DISCLOSURE_MARKERS),
        "default_voice_marker_count": marker_count(text, VOICE_MARKERS),
        "word_count": len(words),
        "repetition_rate": rounded(repetition_rate(text)),
    }


def candidate_features_for_edit(taxonomy: Sequence[Mapping[str, Any]], role_b: str) -> list[int]:
    priority = []
    for row in taxonomy:
        generic = row.get("taxonomy")
        role_tax = str(row.get("role_taxonomy", ""))
        if generic == "model_b_only" or role_tax == f"{role_b}_only":
            priority.append(row)
    if not priority:
        priority = [r for r in taxonomy if r.get("taxonomy") == "asymmetric"]
    priority = sorted(priority, key=lambda r: -float(r.get("model_b_specificity_score") or 0.0))
    n = int(os.environ.get("LAB19_EDIT_FEATURES", "1"))
    return [int(r["feature_id"]) for r in priority[: max(1, n)]]


def causal_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ran = [r for r in rows if r.get("status") == "ran"]
    if not ran:
        return [{"status": "skipped", "condition": "", "n": 0, "note": "No causal rows were run."}]
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ran:
        by_condition[str(row["condition"])].append(row)
    baseline_by_prompt = {
        str(row["prompt_id"]): row for row in ran if row.get("condition") == "baseline"
    }
    out = []
    for condition, sub in sorted(by_condition.items()):
        deltas = defaultdict(list)
        for row in sub:
            base = baseline_by_prompt.get(str(row["prompt_id"]))
            if not base:
                continue
            for key in ("default_voice_marker", "politeness_marker", "hedging_marker", "refusal_marker", "self_situation_marker", "word_count", "repetition_rate"):
                try:
                    deltas[key].append(float(row[key]) - float(base[key]))
                except Exception:
                    pass
        out.append({
            "status": "ran",
            "condition": condition,
            "n": len(sub),
            "mean_default_voice_marker": rounded(safe_fmean(row.get("default_voice_marker") for row in sub)),
            "mean_politeness_marker": rounded(safe_fmean(row.get("politeness_marker") for row in sub)),
            "mean_hedging_marker": rounded(safe_fmean(row.get("hedging_marker") for row in sub)),
            "mean_refusal_marker": rounded(safe_fmean(row.get("refusal_marker") for row in sub)),
            "delta_default_voice_marker_vs_baseline": rounded(safe_fmean(deltas["default_voice_marker"])),
            "delta_politeness_marker_vs_baseline": rounded(safe_fmean(deltas["politeness_marker"])),
            "delta_hedging_marker_vs_baseline": rounded(safe_fmean(deltas["hedging_marker"])),
            "delta_refusal_marker_vs_baseline": rounded(safe_fmean(deltas["refusal_marker"])),
            "delta_word_count_vs_baseline": rounded(safe_fmean(deltas["word_count"])),
            "delta_repetition_rate_vs_baseline": rounded(safe_fmean(deltas["repetition_rate"])),
        })
    return out


def run_optional_causal_validation(
    ctx: bench.RunContext,
    bundle_b: bench.ModelBundle,
    model: PairedCrosscoder,
    taxonomy: Sequence[Mapping[str, Any]],
    stats: Mapping[str, Any],
    depth_b: int,
    role_b: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import torch

    candidates = candidate_features_for_edit(taxonomy, role_b)
    if not candidates:
        rows = [{
            "status": "skipped_no_candidate_feature",
            "feature_id": "",
            "condition": "",
            "prompt_id": "",
            "generation": "",
            "note": "No model-B-skewed or asymmetric features were available.",
        }]
        return rows, causal_summary_rows(rows), {"status": "skipped_no_candidate_feature"}

    layer = max(0, min(bundle_b.anatomy.n_layers - 1, depth_b - 1))
    prompts = benign_causal_prompts(bundle_b)
    std_b = stats["std_b"].reshape(-1)
    dec_b = model.D_b * std_b.reshape(1, -1)
    g = torch.Generator(device="cpu").manual_seed(int(ctx.args.seed) + 773)
    rows: list[dict[str, Any]] = []
    manifest = {
        "status": "ran",
        "candidate_features": candidates,
        "stream_depth_b": depth_b,
        "injection_layer": layer,
        "dose": STEERING_DOSE,
        "safety_scope": "benign prompts only; no refusal ablation; activation addition only",
        "score_warning": "marker columns are automatic heuristics; hand_label_behavior is intentionally blank for students.",
    }

    for fid in candidates:
        vec = dec_b[fid].detach().float().cpu()
        vec = vec / vec.norm().clamp_min(1e-9)
        random_vec = torch.randn(vec.shape, generator=g)
        random_vec = random_vec / random_vec.norm().clamp_min(1e-9)
        conditions = [
            ("baseline", None),
            ("feature_plus", (layer, vec, STEERING_DOSE)),
            ("feature_plus_low", (layer, vec, STEERING_DOSE / 2)),
            ("feature_minus", (layer, vec, -STEERING_DOSE)),
            ("random_plus", (layer, random_vec, STEERING_DOSE)),
            ("random_minus", (layer, random_vec, -STEERING_DOSE)),
        ]
        for condition, steer in conditions:
            outs = bench.generate_continuous(
                bundle_b,
                [p for _, p in prompts],
                MAX_NEW_TOKENS,
                max_concurrent=ENGINE_MAX_CONCURRENT,
                progress_label=f"lab19 causal f{fid} {condition}",
                steer=steer,
            )
            for (prompt_id, prompt), text in zip(prompts, outs):
                scores = score_generation(text)
                rows.append({
                    "status": "ran",
                    "feature_id": fid,
                    "condition": condition,
                    "stream_depth_b": depth_b,
                    "injection_layer": layer,
                    "dose": 0.0 if steer is None else steer[2],
                    "prompt_id": prompt_id,
                    **scores,
                    "generation": text.replace("\n", "\\n"),
                    "prompt_excerpt": prompt[-220:].replace("\n", "\\n"),
                    "hand_label_behavior": "",
                    "hand_label_note": "",
                })
    summary = causal_summary_rows(rows)
    # Add a coarse verdict based on specificity over random. This is deliberately
    # weak and marker-based; the handout requires hand labels for a defended claim.
    by_cond = {r["condition"]: r for r in summary if r.get("status") == "ran"}
    fp = by_cond.get("feature_plus", {})
    rp = by_cond.get("random_plus", {})
    try:
        specificity = float(fp.get("delta_default_voice_marker_vs_baseline") or 0.0) - float(rp.get("delta_default_voice_marker_vs_baseline") or 0.0)
    except Exception:
        specificity = float("nan")
    manifest["default_voice_specificity_over_random"] = rounded(specificity)
    manifest["marker_based_verdict"] = (
        "candidate_behavioral_handle" if math.isfinite(specificity) and specificity >= 0.20 else "not_validated_by_marker_control"
    )
    return rows, summary, manifest


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_taxonomy_counts(ctx: bench.RunContext, taxonomy: Sequence[Mapping[str, Any]]) -> None:
    counts = Counter(str(r["role_taxonomy"]) for r in taxonomy)
    labels = sorted(counts)
    fig, ax = bench.new_figure(figsize=(8.2, 4.6))
    ax.bar(labels, [counts[l] for l in labels])
    ax.set_ylabel("feature count")
    ax.set_title("Crosscoder feature taxonomy")
    ax.tick_params(axis="x", rotation=35)
    bench.style_ax(ax, legend=False)
    bench.save_figure(ctx, fig, "feature_taxonomy_counts.png", "Counts of shared, model-specific, asymmetric, and dead crosscoder features.")


def plot_exclusivity(ctx: bench.RunContext, taxonomy: Sequence[Mapping[str, Any]], random_rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(8.4, 5.0))
    vals = [float(r["model_b_activation_share"]) for r in taxonomy if isinstance(r.get("model_b_activation_share"), (int, float))]
    rand = [float(r["model_b_activation_share"]) for r in random_rows if isinstance(r.get("model_b_activation_share"), (int, float))]
    if rand:
        ax.hist(rand, bins=18, alpha=0.35, label="random directions")
    if vals:
        ax.hist(vals, bins=18, alpha=0.75, label="crosscoder features")
    ax.axvline(0.5, linestyle="--", linewidth=1)
    ax.set_xlabel("model B activation share")
    ax.set_ylabel("feature count")
    ax.set_title("Feature exclusivity versus random-direction baseline")
    bench.style_ax(ax, legend=True)
    bench.save_figure(ctx, fig, "feature_exclusivity_histogram.png", "Histogram of feature model-B activation shares with random-direction baseline.")


def plot_crosscoder_reconstruction(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    labels = ["train A", "train B", "eval A", "eval B"]
    vals = [
        metrics.get("train_fvu_model_a"),
        metrics.get("train_fvu_model_b"),
        metrics.get("eval_fvu_model_a"),
        metrics.get("eval_fvu_model_b"),
    ]
    numeric = [float(v) if isinstance(v, (int, float)) or (isinstance(v, str) and v) else float("nan") for v in vals]
    fig, ax = bench.new_figure(figsize=(7.4, 4.5))
    ax.bar(labels, numeric)
    ax.set_ylabel("FVU, lower is better")
    ax.set_title("Crosscoder reconstruction quality")
    bench.style_ax(ax, legend=False)
    bench.save_figure(ctx, fig, "crosscoder_reconstruction.png", "Train/eval reconstruction FVU for both sides of the paired crosscoder.")


def plot_template_control(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    top = [r for r in rows if isinstance(r.get("abs_template_gap"), (int, float))][:16]
    if not top:
        return
    fig, ax = bench.new_figure(figsize=(8.6, 5.0))
    labels = [str(r["feature_id"]) for r in top]
    vals = [float(r["mean_compare_chat_minus_raw_activation_model_b"] or 0.0) for r in top]
    ax.bar(labels, vals)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("feature id")
    ax.set_ylabel("chat minus raw activation, model B")
    ax.set_title("Template-control gaps for top affected features")
    bench.style_ax(ax, legend=False)
    bench.save_figure(ctx, fig, "template_control_gaps.png", "Features most sensitive to chat-template rendering in the comparison model.")


def plot_direction_bridge(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    computed = [r for r in rows if r.get("status") == "computed" and isinstance(r.get("abs_cosine"), (int, float))]
    fig, ax = bench.new_figure(figsize=(8.0, 4.4))
    if computed:
        top = sorted(computed, key=lambda r: -float(r["abs_cosine"]))[:16]
        labels = [f"f{r['feature_id']}" for r in top]
        vals = [float(r["cosine_with_model_b_decoder"]) for r in top]
        ax.bar(labels, vals)
        ax.axhline(0, linewidth=1)
        ax.set_ylabel("cosine with bridge direction")
        ax.set_title("Feature decoder bridge to prior-lab direction")
    else:
        ax.axis("off")
        ax.text(0.02, 0.68, "Direction bridge not configured.", fontsize=12)
        ax.text(0.02, 0.46, "Set LAB19_BRIDGE_STATE to a prior-lab state .pt file.", fontsize=10)
        ax.text(0.02, 0.25, "The table still records the expected residual-space convention.", fontsize=10)
    bench.style_ax(ax, legend=False)
    bench.save_figure(ctx, fig, "feature_direction_bridge.png", "Bridge between model-diff feature decoders and saved prior-lab directions.")


def plot_causal_validation(ctx: bench.RunContext, summary: Sequence[Mapping[str, Any]]) -> None:
    rows = [r for r in summary if r.get("status") == "ran"]
    if not rows:
        return
    fig, ax = bench.new_figure(figsize=(8.5, 4.7))
    labels = [str(r["condition"]) for r in rows]
    vals = [float(r.get("delta_default_voice_marker_vs_baseline") or 0.0) for r in rows]
    ax.bar(labels, vals)
    ax.axhline(0, linewidth=1)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("default-voice marker delta vs baseline")
    ax.set_title("Optional feature intervention, marker-based smoke score")
    bench.style_ax(ax, legend=False)
    bench.save_figure(ctx, fig, "causal_feature_validation.png", "Optional benign feature-intervention effects against random-feature controls.")


# ---------------------------------------------------------------------------
# Reports and ledger
# ---------------------------------------------------------------------------


def audit_status(metrics: Mapping[str, Any]) -> str:
    identity = bool(metrics.get("identity_pair"))
    counts = metrics.get("taxonomy_counts", {}) or {}
    role_counts = metrics.get("role_taxonomy_counts", {}) or {}
    n_features = int(metrics.get("n_features") or 0)
    nonshared = n_features - int(counts.get("shared", 0) or 0) - int(counts.get("dead", 0) or 0)
    try:
        eval_fvu = max(float(metrics.get("eval_fvu_model_a") or 0.0), float(metrics.get("eval_fvu_model_b") or 0.0))
    except Exception:
        eval_fvu = float("nan")
    template_warning = int(metrics.get("n_template_dominated_model_b_features") or 0)
    if identity and n_features and nonshared / n_features > 0.35:
        return "identity_pair_failed_or_dictionary_unstable"
    if math.isfinite(eval_fvu) and eval_fvu > 0.75:
        return "weak_reconstruction"
    if template_warning > max(2, 0.25 * n_features):
        return "template_control_dominates"
    if int(counts.get("model_b_only", 0) or 0) > 0 or any("instruct_only" in str(k) and int(v) > 0 for k, v in role_counts.items()):
        return "candidate_model_diff_features_with_audit_caveats"
    return "mostly_shared_or_inconclusive"


def write_card(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    status = metrics.get("audit_status")
    lines = [
        "# Lab 19 Model Diffing Card",
        "",
        "## Verdict",
        "",
        f"- audit status: `{status}`",
        f"- model A: `{metrics.get('model_a')}` as `{metrics.get('model_a_role')}`",
        f"- model B: `{metrics.get('model_b')}` as `{metrics.get('model_b_role')}`",
        f"- identity-pair smoke run: `{metrics.get('identity_pair')}`",
        f"- stream depths: A={metrics.get('depth_a')}, B={metrics.get('depth_b')}",
        f"- crosscoder features: {metrics.get('n_features')}",
        f"- eval FVU A/B: {metrics.get('eval_fvu_model_a')} / {metrics.get('eval_fvu_model_b')}",
        f"- taxonomy counts: `{metrics.get('taxonomy_counts')}`",
        "",
        "## Claim posture",
        "",
        "The feature taxonomy is `DECODE/ATTR`: it is a sparse coordinate system for a model-pair difference under this prompt inventory. It is not a proof that instruction following, alignment, or a real assistant identity lives in a feature.",
        "",
        "A `model_b_only` or `instruct_only` feature becomes a serious candidate only if it survives the template, norm, prompt-family, random-direction, and held-out-family checks. The optional feature intervention can add a narrow behavioral-handle claim on benign prompts, with hand labels required before it enters the ledger as more than a smoke test.",
        "",
        "## Read next",
        "",
        "1. `diagnostics/model_pair.json`",
        "2. `tables/feature_taxonomy.csv`",
        "3. `tables/template_control_summary.csv`",
        "4. `tables/instruct_only_feature_gallery.csv`",
        "5. `operationalization_audit.md`",
        "",
    ]
    path = ctx.path("model_diffing_card.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Read-first card for Lab 19 model diffing.")


def write_report(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 19 Model Diffing Report",
        "",
        "## Run identity",
        "",
        f"- Model A: `{metrics.get('model_a')}` ({metrics.get('model_a_role')})",
        f"- Model B: `{metrics.get('model_b')}` ({metrics.get('model_b_role')})",
        f"- Prompt rows: {metrics.get('n_prompts')}",
        f"- Stream depths: A={metrics.get('depth_a')}, B={metrics.get('depth_b')}",
        f"- Features: {metrics.get('n_features')}",
        f"- Identity-pair smoke: {metrics.get('identity_pair')}",
        "",
        "## Reconstruction",
        "",
        f"- Train FVU A/B: {metrics.get('train_fvu_model_a')} / {metrics.get('train_fvu_model_b')}",
        f"- Eval FVU A/B: {metrics.get('eval_fvu_model_a')} / {metrics.get('eval_fvu_model_b')}",
        f"- Pair feature density: {metrics.get('mean_pair_feature_density')}",
        "",
        "## Feature taxonomy",
        "",
        f"- Generic taxonomy counts: `{metrics.get('taxonomy_counts')}`",
        f"- Role taxonomy counts: `{metrics.get('role_taxonomy_counts')}`",
        f"- Template-dominated model-B features: {metrics.get('n_template_dominated_model_b_features')}",
        f"- Random-direction model-B-specific baseline rate: {metrics.get('random_baseline_model_b_specific_rate')}",
        "",
        "## Optional causal validation",
        "",
        f"- Status counts: `{metrics.get('causal_validation_status')}`",
        f"- Marker-based verdict: `{metrics.get('causal_marker_verdict')}`",
        "",
        "## Bottom line",
        "",
        f"Audit status: `{metrics.get('audit_status')}`. Read `operationalization_audit.md` before naming any feature as assistant voice, alignment, refusal, sycophancy, or personality.",
        "",
    ]
    path = ctx.path("model_diffing_report.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Human-readable Lab 19 model-diffing report.")


def write_run_summary(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 19 Run Summary: Model Diffing With Crosscoders",
        "",
        f"- model A: `{metrics.get('model_a')}` ({metrics.get('model_a_role')})",
        f"- model B: `{metrics.get('model_b')}` ({metrics.get('model_b_role')})",
        f"- identity-pair smoke: {metrics.get('identity_pair')}",
        f"- prompt rows: {metrics.get('n_prompts')}",
        f"- selected stream depths: A={metrics.get('depth_a')}, B={metrics.get('depth_b')}",
        f"- features: {metrics.get('n_features')}",
        f"- eval FVU A/B: {metrics.get('eval_fvu_model_a')} / {metrics.get('eval_fvu_model_b')}",
        f"- taxonomy counts: `{metrics.get('role_taxonomy_counts')}`",
        f"- audit status: `{metrics.get('audit_status')}`",
        "",
        "Start with `model_diffing_card.md`, then inspect the feature taxonomy, the template-control table, and the top-context gallery. The plot is a lantern, not a verdict.",
        "",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Standard Lab 19 run summary.")


def write_operationalization_audit(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 19 Operationalization Audit",
        "",
        "## Favorite story",
        "",
        "The tempting story is that model-B-only or instruct-only crosscoder features are pieces of instruction following, alignment, assistant voice, or personality.",
        "",
        "## What was actually measured",
        "",
        "A small paired sparse dictionary was trained on matched final-token residual activations from two models on one prompt inventory. Feature labels are hypotheses about coordinate usage, not entity-realism claims.",
        "",
        "## Cheap explanations and artifacts that attack them",
        "",
        "| cheap explanation | required artifact | what would deflate the story |",
        "|---|---|---|",
        "| chat-template token residue | `tables/template_control_summary.csv` | the feature mostly fires on `compare_chat` variants |",
        "| prompt-family imbalance | `tables/feature_context_gallery.csv` and `tables/prompt_inventory.csv` | top contexts are one source or one family |",
        "| activation-norm shift | `diagnostics/activation_norms.csv` and `tables/activation_norm_controls.csv` | model-B norms are globally larger in the same families |",
        "| crosscoder artifact | `tables/random_feature_baseline.csv` | random directions look equally model-specific |",
        "| shallow output-format habit | `tables/default_voice_marker_rates.csv` and optional `tables/causal_feature_validation.csv` | marker behavior tracks style words, not feature specificity |",
        "| direction-name overreach | `tables/feature_direction_bridge.csv` | saved persona/agreement/certainty directions do not align, or align only through template features |",
        "",
        "## Current run readings",
        "",
        f"- audit status: `{metrics.get('audit_status')}`",
        f"- identity-pair smoke: `{metrics.get('identity_pair')}`",
        f"- train/eval FVU A: {metrics.get('train_fvu_model_a')} / {metrics.get('eval_fvu_model_a')}",
        f"- train/eval FVU B: {metrics.get('train_fvu_model_b')} / {metrics.get('eval_fvu_model_b')}",
        f"- template-dominated model-B features: {metrics.get('n_template_dominated_model_b_features')}",
        f"- random baseline model-B-specific rate: {metrics.get('random_baseline_model_b_specific_rate')}",
        "",
        "## Allowed claim",
        "",
        "Allowed by default: a feature-level model-diff handle under this model pair, site, depth, and prompt inventory. Mechanism language requires a specific intervention that beats random and style controls. Assistant-voice or alignment language requires the template and prompt-family controls to stop being the best explanation.",
        "",
    ]
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "audit", "Operationalization limits and cheap-explanation audit for Lab 19.")


def write_labeling_guide(ctx: bench.RunContext) -> None:
    lines = [
        "# Lab 19 Feature Gallery Labeling Guide",
        "",
        "For each selected feature, inspect the top contexts and assign the narrowest label that survives counterexamples.",
        "",
        "Recommended label axes:",
        "",
        "- `template`: fires on chat scaffolding, role markers, generation prompts, or boilerplate.",
        "- `refusal_boundary`: private information, refusal, inability, or boundary-setting contexts.",
        "- `politeness`: please/thanks/softening/formal courtesy.",
        "- `hedging`: uncertainty, caveats, verification, or probability language.",
        "- `disclosure`: source limits, transparency, or epistemic caution.",
        "- `self_situation`: assistant or language-model self-description.",
        "- `default_voice`: a broader assistant-tone hypothesis only after template and marker controls look weak.",
        "- `topic`: a domain or dataset topic rather than a model-role feature.",
        "- `dead_or_artifact`: no coherent context pattern.",
        "",
        "Write one counterexample. A feature without a counterexample has probably not been audited yet.",
        "",
    ]
    path = ctx.path("tables", "feature_labeling_guide.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "guide", "Student labeling guide for crosscoder feature galleries.")


# ---------------------------------------------------------------------------
# Main lab entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    import torch

    args = ctx.args
    compare_id, compare_revision, compare_source = comparison_model_spec(ctx, bundle)
    same_model = compare_id == bundle.anatomy.model_id and (compare_revision or bundle.anatomy.revision) == bundle.anatomy.revision
    role_a = infer_model_role(bundle.anatomy.model_id, is_compare=False)
    role_b = infer_model_role(compare_id, is_compare=True)

    compare_tok = bundle.tokenizer if same_model else comparison_tokenizer(compare_id, compare_revision, ctx)
    prompts, prompt_info = load_prompt_inventory(args, compare_tok)
    split = make_split(prompts, int(args.seed))

    prompt_path = ctx.path("tables", "prompt_inventory.csv")
    bench.write_csv_with_context(ctx, prompt_path, [dataclasses.asdict(p) for p in prompts])
    ctx.register_artifact(prompt_path, "table", "Prompt inventory for matched model-pair activation collection.")

    split_path = ctx.path("diagnostics", "split_audit.csv")
    bench.write_csv_with_context(ctx, split_path, split_rows(prompts, split))
    ctx.register_artifact(split_path, "diagnostic", "Prompt-group train/eval split for crosscoder training and stability checks.")

    manifest_path = ctx.path("diagnostics", "frozen_prompt_manifest.json")
    bench.write_json(manifest_path, prompt_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Prompt inventory source, hash, family counts, and selection rule.")

    depth_a = select_depth(bundle, "a")

    # Self-check and capture model A before loading model B. Full science pairs
    # can be too large to keep both models resident on a single Tier-B GPU.
    run_prefixed_hook_parity_check(ctx, bundle, prompts[0].text, "model_a")
    run_prefixed_lens_self_check(ctx, bundle, prompts[0].text, "model_a")
    x_a, rows_a_raw = collect_model_activations(ctx, bundle, prompts, depth_a, "model_a")

    if same_model:
        compare_bundle = bundle
        depth_b = depth_a
        x_b = x_a.clone()
        rows_b_raw = [{**row, "model_prefix": "model_b", "note": "reused identity-pair activation from model_a"} for row in rows_a_raw]
        reuse_payload = {"model_prefix": "model_b", "reused_model_a": True, "ok": True, "reason": "identity_pair"}
        path = ctx.path("diagnostics", "model_b_hook_parity.json")
        bench.write_json(path, reuse_payload)
        ctx.register_artifact(path, "diagnostic", "Model B hook parity reused from identity-pair model A check.")
        path = ctx.path("diagnostics", "model_b_logit_lens_self_check.json")
        bench.write_json(path, reuse_payload)
        ctx.register_artifact(path, "diagnostic", "Model B lens check reused from identity-pair model A check.")
    else:
        release_info = maybe_release_primary_model(ctx, bundle, same_model=False)
        release_path = ctx.path("diagnostics", "primary_model_memory_release.json")
        bench.write_json(release_path, release_info)
        ctx.register_artifact(release_path, "diagnostic", "Best-effort primary-model memory release before loading comparison model.")
        compare_bundle = load_comparison_bundle(ctx, compare_id, compare_revision)
        depth_b = select_depth(compare_bundle, "b") if os.environ.get("LAB19_DEPTH_B") or os.environ.get("LAB19_DEPTH") else max(1, min(compare_bundle.anatomy.n_layers, int(round(compare_bundle.anatomy.n_layers * 0.65))))
        run_prefixed_hook_parity_check(ctx, compare_bundle, prompts[0].text, "model_b")
        run_prefixed_lens_self_check(ctx, compare_bundle, prompts[0].text, "model_b")
        x_b, rows_b_raw = collect_model_activations(ctx, compare_bundle, prompts, depth_b, "model_b")

    pair_info = {
        "model_a": bundle.anatomy.model_id,
        "model_b": compare_bundle.anatomy.model_id,
        "model_a_revision": bundle.anatomy.revision,
        "model_b_revision": compare_bundle.anatomy.revision,
        "model_a_role": role_a,
        "model_b_role": role_b,
        "comparison_model_source": compare_source,
        "identity_pair": bool(same_model),
        "depth_a": depth_a,
        "depth_b": depth_b,
        "d_model_a": bundle.anatomy.d_model,
        "d_model_b": compare_bundle.anatomy.d_model,
        "prompt_inventory": prompt_info,
        "stream_convention": "streams[k] is pre-norm residual after k blocks; steering at layer k writes into streams[k + 1].",
        "tier_a_note": "Identity-pair or tiny-model smoke runs prove plumbing, not science.",
    }
    pair_path = ctx.path("diagnostics", "model_pair.json")
    bench.write_json(pair_path, pair_info)
    ctx.register_artifact(pair_path, "diagnostic", "Model-pair metadata, selected stream depths, and prompt-inventory summary.")

    acts = pair_activations_from_parts(x_a, rows_a_raw, x_b, rows_b_raw, depth_a, depth_b, split)
    act_path = ctx.path("diagnostics", "activation_norms.csv")
    bench.write_csv_with_context(ctx, act_path, acts.prompt_rows)
    ctx.register_artifact(act_path, "diagnostic", "Prompt-level token counts and residual norm controls for the model pair.")

    # Grouped norm controls are easier to read than one long diagnostic table.
    # Reconstruct pseudo per-model rows from the joined table for the aggregate helper.
    pseudo_a = [{"row_index": r["row_index"], "family": r["family"], "variant": r["variant"], "residual_norm": r["norm_model_a"], "n_tokens": r["n_tokens_model_a"]} for r in acts.prompt_rows]
    pseudo_b = [{"row_index": r["row_index"], "family": r["family"], "variant": r["variant"], "residual_norm": r["norm_model_b"], "n_tokens": r["n_tokens_model_b"]} for r in acts.prompt_rows]
    norm_summary = activation_norm_control_rows(pseudo_a, pseudo_b)
    norm_summary_path = ctx.path("tables", "activation_norm_controls.csv")
    bench.write_csv_with_context(ctx, norm_summary_path, norm_summary)
    ctx.register_artifact(norm_summary_path, "table", "Grouped activation norm and token-count controls by family and prompt variant.")

    crosscoder, train_metrics, stats, curve = train_crosscoder(ctx, acts, int(args.seed))
    curve_path = ctx.path("tables", "crosscoder_training_curve.csv")
    bench.write_csv_with_context(ctx, curve_path, curve)
    ctx.register_artifact(curve_path, "table", "Crosscoder training curve with train/eval reconstruction FVU and feature density.")

    taxonomy = feature_taxonomy_rows(crosscoder, stats, prompts, split, role_a, role_b)
    taxonomy_path = ctx.path("tables", "feature_taxonomy.csv")
    bench.write_csv_with_context(ctx, taxonomy_path, taxonomy)
    ctx.register_artifact(taxonomy_path, "table", "Crosscoder feature taxonomy with model-specificity, decoder norms, and audit flags.")
    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, taxonomy)
    ctx.register_artifact(results_path, "results", "Alias of feature_taxonomy.csv for the standard run contract.")

    stability = feature_stability_rows(taxonomy)
    stability_path = ctx.path("tables", "feature_eval_stability.csv")
    bench.write_csv_with_context(ctx, stability_path, stability)
    ctx.register_artifact(stability_path, "table", "Feature train/eval activity and concentration warnings.")

    gallery = gallery_rows(taxonomy, stats, prompts)
    gallery_path = ctx.path("tables", "feature_context_gallery.csv")
    bench.write_csv_with_context(ctx, gallery_path, gallery)
    ctx.register_artifact(gallery_path, "table", "Top prompt contexts for selected model-specific and asymmetric crosscoder features.")
    # Backward-compatible alias for early Lab 19 drafts that centered the
    # base-vs-instruct comparison. The role-neutral file above is the preferred
    # artifact for non-instruct comparison pairs.
    legacy_gallery_path = ctx.path("tables", "instruct_only_feature_gallery.csv")
    bench.write_csv_with_context(ctx, legacy_gallery_path, gallery)
    ctx.register_artifact(legacy_gallery_path, "table", "Backward-compatible alias of feature_context_gallery.csv.")
    write_labeling_guide(ctx)

    template_rows = template_control_summary(taxonomy, stats, prompts)
    template_path = ctx.path("tables", "template_control_summary.csv")
    bench.write_csv_with_context(ctx, template_path, template_rows)
    ctx.register_artifact(template_path, "table", "Raw-vs-chat template control gaps for model-B feature activations.")

    marker_path = ctx.path("tables", "default_voice_marker_rates.csv")
    bench.write_csv_with_context(ctx, marker_path, voice_marker_rows(prompts))
    ctx.register_artifact(marker_path, "table", "Prompt-text default-assistant, politeness, hedging, and refusal marker controls.")

    random_rows = random_feature_baseline_rows(acts, int(args.seed))
    random_path = ctx.path("tables", "random_feature_baseline.csv")
    bench.write_csv_with_context(ctx, random_path, random_rows)
    ctx.register_artifact(random_path, "table", "Random-direction exclusivity baseline for model-specific feature claims.")

    bridge = direction_bridge_rows(taxonomy, crosscoder, stats, compare_bundle)
    bridge_path = ctx.path("tables", "feature_direction_bridge.csv")
    bench.write_csv_with_context(ctx, bridge_path, bridge)
    ctx.register_artifact(bridge_path, "table", "Feature-decoder cosines to saved prior-lab directions when LAB19_BRIDGE_STATE is set.")

    if getattr(args, "run_edit", False):
        causal_rows, causal_summary, causal_manifest = run_optional_causal_validation(ctx, compare_bundle, crosscoder, taxonomy, stats, depth_b, role_b)
    else:
        causal_rows = [{
            "status": "skipped",
            "feature_id": "",
            "condition": "",
            "prompt_id": "",
            "generation": "",
            "note": "Rerun Lab 19 with --run-edit to perform the optional benign feature-intervention smoke test.",
        }]
        causal_summary = causal_summary_rows(causal_rows)
        causal_manifest = {"status": "skipped", "note": "--run-edit was not passed."}

    causal_path = ctx.path("tables", "causal_feature_validation.csv")
    bench.write_csv_with_context(ctx, causal_path, causal_rows)
    ctx.register_artifact(causal_path, "table", "Optional benign feature-intervention generations and marker scores.")
    causal_summary_path = ctx.path("tables", "causal_feature_validation_summary.csv")
    bench.write_csv_with_context(ctx, causal_summary_path, causal_summary)
    ctx.register_artifact(causal_summary_path, "table", "Condition-level summary of optional feature-intervention marker effects.")
    causal_manifest_path = ctx.path("diagnostics", "causal_feature_validation_manifest.json")
    bench.write_json(causal_manifest_path, causal_manifest)
    ctx.register_artifact(causal_manifest_path, "diagnostic", "Scope, candidate features, and verdict for optional causal validation.")

    state = {
        "model_a": bundle.anatomy.model_id,
        "model_b": compare_bundle.anatomy.model_id,
        "model_a_role": role_a,
        "model_b_role": role_b,
        "depth_a": depth_a,
        "depth_b": depth_b,
        "crosscoder_type": "paired_sparse_crosscoder_with_shared_feature_ids",
        "crosscoder": {
            "W_a": crosscoder.W_a.detach().cpu(),
            "W_b": crosscoder.W_b.detach().cpu(),
            "b": crosscoder.b.detach().cpu(),
            "D_a": crosscoder.D_a.detach().cpu(),
            "D_b": crosscoder.D_b.detach().cpu(),
        },
        "normalization": {
            "mean_a": stats["mean_a"].detach().cpu(),
            "std_a": stats["std_a"].detach().cpu(),
            "mean_b": stats["mean_b"].detach().cpu(),
            "std_b": stats["std_b"].detach().cpu(),
        },
        "feature_taxonomy": taxonomy,
        "prompt_inventory": [dataclasses.asdict(p) for p in prompts],
    }
    state_path = ctx.path("state", "crosscoder_state.pt")
    torch.save(state, state_path)
    ctx.register_artifact(state_path, "tensor", "Trained paired crosscoder weights, normalization, taxonomy, and prompt inventory.")

    metadata = {
        "lab_id": LAB_ID,
        "crosscoder_type": "paired_sparse_crosscoder_with_shared_feature_ids",
        "model_a": bundle.anatomy.model_id,
        "model_b": compare_bundle.anatomy.model_id,
        "depth_a": depth_a,
        "depth_b": depth_b,
        "feature_decoder_space": "D_a and D_b are in normalized activation units; multiply by std_a/std_b for residual units.",
        "stream_convention": pair_info["stream_convention"],
        "training": train_metrics,
    }
    metadata_path = ctx.path("state", "crosscoder_metadata.json")
    bench.write_json(metadata_path, metadata)
    ctx.register_artifact(metadata_path, "state", "Human-readable crosscoder state metadata and stream convention.")

    taxonomy_counts = dict(Counter(row["taxonomy"] for row in taxonomy))
    role_taxonomy_counts = dict(Counter(row["role_taxonomy"] for row in taxonomy))
    template_dominated = [r for r in template_rows if r.get("template_dominated_warning")]
    model_b_specific_random_rate = safe_fmean(float(r.get("would_look_model_b_specific") is True) for r in random_rows)
    metrics = {
        **pair_info,
        **train_metrics,
        "n_prompts": len(prompts),
        "taxonomy_counts": taxonomy_counts,
        "role_taxonomy_counts": role_taxonomy_counts,
        "n_gallery_rows": len(gallery),
        "n_template_dominated_model_b_features": len(template_dominated),
        "random_baseline_model_b_specific_rate": rounded(model_b_specific_random_rate),
        "causal_validation_status": dict(Counter(row.get("status", "") for row in causal_rows)),
        "causal_marker_verdict": causal_manifest.get("marker_based_verdict", causal_manifest.get("status")),
    }
    metrics["audit_status"] = audit_status(metrics)
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 19 metrics and dynamic audit status.")

    if not getattr(args, "no_plots", False):
        plot_taxonomy_counts(ctx, taxonomy)
        plot_exclusivity(ctx, taxonomy, random_rows)
        plot_crosscoder_reconstruction(ctx, metrics)
        plot_template_control(ctx, template_rows)
        plot_direction_bridge(ctx, bridge)
        plot_causal_validation(ctx, causal_summary)

    write_card(ctx, metrics)
    write_report(ctx, metrics)
    write_run_summary(ctx, metrics)
    write_operationalization_audit(ctx, metrics)

    run_name = ctx.run_dir.name
    causal_tag = "CAUSAL" if causal_manifest.get("marker_based_verdict") == "candidate_behavioral_handle" else "CAUSAL?"
    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "DECODE/ATTR",
            "text": (
                f"At residual depths A={depth_a}, B={depth_b}, a paired crosscoder over {len(prompts)} matched prompts found "
                f"role-taxonomy counts {role_taxonomy_counts} with eval FVU {metrics.get('eval_fvu_model_a')} / {metrics.get('eval_fvu_model_b')}. "
                f"Audit status: {metrics['audit_status']}. This is a model-pair feature taxonomy under the sampled prompt distribution, not a mechanism claim."
            ),
            "artifact": f"runs/{run_name}/tables/feature_taxonomy.csv",
            "falsifier": "Template controls, norm controls, random-direction baselines, or a held-out prompt inventory erase the model-specificity pattern.",
        },
        {
            "id": f"{LAB_ID}-C2",
            "tag": causal_tag,
            "text": (
                f"Optional benign feature-intervention status is {causal_manifest.get('status')} with marker verdict "
                f"{causal_manifest.get('marker_based_verdict', 'not_run')}. Treat this as a causal handle only if feature-plus beats random controls and hand labels agree."
            ),
            "artifact": f"runs/{run_name}/tables/causal_feature_validation_summary.csv",
            "falsifier": "Random-feature intervention matches the effect, hand labels reject marker scoring, or the generation change is verbosity/refusal rather than the intended behavior.",
        },
        {
            "id": f"{LAB_ID}-C3",
            "tag": "DECODE/ATTR audited",
            "text": (
                f"Default-assistant-voice language is not licensed unless candidate {role_b}-only features survive `template_control_summary.csv`, "
                "prompt-family concentration checks, and the feature gallery labels. The current run's audit status is "
                f"{metrics['audit_status']}."
            ),
            "artifact": f"runs/{run_name}/operationalization_audit.md",
            "falsifier": "The apparent voice feature fires mainly on chat-template variants, one prompt source, or shallow marker words.",
        },
    ]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
