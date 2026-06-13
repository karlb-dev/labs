"""Lab 19: Model diffing with a small paired crosscoder.

The full science target is OLMo base versus OLMo instruct. This first course
implementation keeps the method runnable and inspectable:

* collect matched residual-stream activations from a model pair;
* train a small paired sparse crosscoder in plain PyTorch;
* classify features as shared, base-skewed, instruct-skewed, asymmetric, or dead;
* build top-context galleries and cheap-control audits;
* optionally run a narrow activation-addition validation for one instruct-skewed
  feature with ``--run-edit``.

Evidence labels:
  * DECODE/ATTR for feature taxonomy and top-activation galleries;
  * CAUSAL only for the optional controlled feature intervention.

The lab is intentionally modest about its claims. A crosscoder feature is a
handle on a model-pair difference under a prompt distribution, not proof that
alignment, identity, or "assistant voice" lives in one atom.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import math
import os
import statistics
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import interp_bench as bench

LAB_ID = "L19"

CROSSCODER_FEATURES = 96
TRAIN_STEPS_TIER_A = 160
TRAIN_STEPS_TIER_B = 320
LEARNING_RATE = 2e-3
L1_WEIGHT = 2e-3
GALLERY_FEATURES = 18
GALLERY_CONTEXTS = 6
MAX_NEW_TOKENS = 40
ENGINE_MAX_CONCURRENT = 8

VOICE_MARKERS = (
    "i can", "i'll", "happy to", "glad to", "help", "assist",
    "as an ai", "i cannot", "i can't", "please", "sure",
)


@dataclasses.dataclass
class PromptItem:
    prompt_id: str
    family: str
    source: str
    variant: str
    text: str
    note: str = ""


@dataclasses.dataclass
class PairActivations:
    prompt_rows: list[dict[str, Any]]
    x_a: Any
    x_b: Any
    depth_a: int
    depth_b: int


def stable_hash_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def rounded(x: Any, ndigits: int = 4) -> Any:
    try:
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            return round(float(x), ndigits)
    except Exception:
        pass
    return x


def none_if_nan(x: Any, ndigits: int = 4) -> Any:
    try:
        val = float(x)
    except Exception:
        return x
    if not math.isfinite(val):
        return None
    return round(val, ndigits)


def safe_fmean(vals: Sequence[float], default: float = float("nan")) -> float:
    finite = []
    for value in vals:
        try:
            f = float(value)
        except Exception:
            continue
        if math.isfinite(f):
            finite.append(f)
    return float(statistics.fmean(finite)) if finite else default


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
    return sum((x - mx) * (y - my) for x, y in pairs) / (dx * dy)


def data_path(name: str) -> Any:
    return bench.COURSE_ROOT / "data" / name


def decode_cell(text: str) -> str:
    return str(text).replace("\\n", "\n")


def tokenize_len(bundle: bench.ModelBundle, text: str) -> int:
    ids = bundle.tokenizer(text, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def supports_chat_template(bundle: bench.ModelBundle) -> bool:
    return bool(getattr(bundle.tokenizer, "chat_template", None))


def render_for_compare_chat(compare_bundle: bench.ModelBundle, user_message: str) -> str:
    if supports_chat_template(compare_bundle):
        return bench.apply_chat_template(
            compare_bundle,
            user_message,
            system="You are a careful assistant. Keep answers concise and accurate.",
            add_generation_prompt=True,
        )
    return "User: " + user_message + "\nAssistant:"


def add_prompt(items: list[PromptItem], prompt_id: str, family: str, source: str, variant: str, text: str, note: str = "") -> None:
    text = text.strip()
    if not text:
        return
    items.append(PromptItem(prompt_id, family, source, variant, text, note))


def load_prompt_inventory(args: Any, compare_bundle: bench.ModelBundle) -> tuple[list[PromptItem], dict[str, Any]]:
    items: list[PromptItem] = []

    persona_path = data_path("persona_register_pairs.csv")
    if persona_path.exists():
        with persona_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                trait = row["trait"]
                add_prompt(items, f"persona_{row['item_id']}_eval_raw", trait, "persona_register_pairs.csv", "raw", decode_cell(row["eval_prompt"]))
                add_prompt(items, f"persona_{row['item_id']}_pos_raw", trait, "persona_register_pairs.csv", "raw_positive", decode_cell(row["prompt_positive"]))
                add_prompt(items, f"persona_{row['item_id']}_eval_chat", trait, "persona_register_pairs.csv", "compare_chat", render_for_compare_chat(compare_bundle, decode_cell(row["eval_prompt"])))

    syc_path = data_path("sycophancy_pressure_items.csv")
    if syc_path.exists():
        with syc_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                family = "sycophancy_" + row.get("condition", "unknown")
                text = decode_cell(row["user_message"])
                add_prompt(items, f"syco_{row['row_id']}_raw", family, "sycophancy_pressure_items.csv", "raw", text)
                if row.get("condition") in {"neutral", "false_belief", "identity_pressure"}:
                    add_prompt(items, f"syco_{row['row_id']}_chat", family, "sycophancy_pressure_items.csv", "compare_chat", render_for_compare_chat(compare_bundle, text))

    cert_path = data_path("certainty_calibration_items.csv")
    if cert_path.exists():
        with cert_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                question = row["question"]
                add_prompt(items, f"cert_{row['item_id']}_raw", "certainty_" + row["family"], "certainty_calibration_items.csv", "raw", question)

    humor_path = data_path("humor_incongruity_pairs.csv")
    if humor_path.exists():
        with humor_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                text = "Setup: " + row["setup"] + "\nEnding:"
                add_prompt(items, f"humor_{row['item_id']}_raw", "humor_" + row["family"], "humor_incongruity_pairs.csv", "raw", text)

    generic = [
        ("default_voice_help", "assistant_voice", "Please help me organize these notes into next steps."),
        ("default_voice_boundary", "assistant_voice", "Can you share private information from another student's record?"),
        ("default_voice_uncertain", "assistant_voice", "I am not sure whether this claim is true. How should I answer carefully?"),
        ("base_plain_fact", "plain_factual", "The capital of Japan is"),
        ("base_plain_code", "plain_code", "A Python function returns None when"),
    ]
    for prompt_id, family, text in generic:
        add_prompt(items, prompt_id + "_raw", family, "authored_lab19", "raw", text)
        add_prompt(items, prompt_id + "_chat", family, "authored_lab19", "compare_chat", render_for_compare_chat(compare_bundle, text))

    # Deduplicate by text while preserving the first metadata row, then apply a
    # deterministic cap. Mixed prompt families are more useful than contiguous
    # blocks from one CSV.
    dedup: dict[str, PromptItem] = {}
    for item in items:
        dedup.setdefault(item.text, item)
    items = list(dedup.values())
    cap = int(getattr(args, "max_examples", 0) or 0)
    if cap > 0:
        items = sorted(items, key=lambda r: stable_hash_int(r.prompt_id))[:cap]
    else:
        items = sorted(items, key=lambda r: (r.family, r.prompt_id))

    if len(items) < 8:
        raise RuntimeError("Lab 19 needs at least 8 prompts for a meaningful smoke crosscoder.")

    info = {
        "n_prompts": len(items),
        "counts_by_family": dict(Counter(item.family for item in items)),
        "counts_by_variant": dict(Counter(item.variant for item in items)),
        "sources": sorted({item.source for item in items}),
        "cap": cap,
        "selection_rule": "deduplicate by text, then stable-hash cap if --max-examples is positive",
    }
    return items, info


def comparison_model_id(ctx: bench.RunContext) -> tuple[str, str | None]:
    env_model = os.environ.get("LAB19_COMPARE_MODEL")
    env_revision = os.environ.get("LAB19_COMPARE_MODEL_REVISION")
    if env_model:
        return env_model, env_revision
    profile = bench.LAB_PROFILES[ctx.args.lab]
    model = profile.get(f"compare_model_tier_{ctx.args.tier}")
    if not model:
        raise RuntimeError("Lab 19 needs compare_model_tier_* in LAB_PROFILES or LAB19_COMPARE_MODEL.")
    return model, env_revision


def load_comparison_bundle(ctx: bench.RunContext, model_id: str, revision: str | None) -> bench.ModelBundle:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = bench.resolve_device(torch, ctx.args.device)
    dtype = bench.resolve_dtype(torch, ctx.args.dtype, device)
    print(f"[lab19] loading comparison model {model_id!r} (device={device}, dtype={dtype})")
    tok = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=ctx.args.trust_remote_code,
        local_files_only=ctx.args.local_files_only,
    )
    kwargs: dict[str, Any] = {
        "revision": revision,
        "trust_remote_code": ctx.args.trust_remote_code,
        "local_files_only": ctx.args.local_files_only,
    }
    if ctx.args.attn_implementation != "auto":
        kwargs["attn_implementation"] = ctx.args.attn_implementation
    if ctx.args.low_cpu_mem_usage:
        kwargs["low_cpu_mem_usage"] = True
    kwargs["dtype"] = dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model = model.to(device)
    model.eval()
    anatomy, blocks, final_norm, lm_head = bench.resolve_anatomy(model, model_id, revision)
    bundle = bench.ModelBundle(
        model=model,
        tokenizer=tok,
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
    bench.write_json(ctx.path("diagnostics", "compare_model_anatomy.json"), anatomy)
    ctx.register_artifact(ctx.path("diagnostics", "compare_model_anatomy.json"), "diagnostic", "Resolved anatomy for Lab 19 comparison model.")
    return bundle


def selected_depths(ctx: bench.RunContext, bundle_a: bench.ModelBundle, bundle_b: bench.ModelBundle) -> tuple[int, int]:
    env = os.environ.get("LAB19_STREAM_DEPTH")
    max_depth = min(bundle_a.anatomy.n_layers, bundle_b.anatomy.n_layers)
    if env:
        depth = max(1, min(max_depth, int(env)))
    elif ctx.args.tier == "a":
        depth = max(1, min(max_depth, 6))
    else:
        depth = max(1, min(max_depth, 16))
    return depth, depth


def capture_vector(bundle: bench.ModelBundle, prompt: str, depth: int) -> Any:
    cap = bench.run_with_residual_cache(bundle, prompt, add_special_tokens=False)
    if depth >= cap.streams.shape[0]:
        depth = cap.streams.shape[0] - 1
    return cap.streams[depth, -1, :]


def collect_pair_activations(
    ctx: bench.RunContext,
    bundle_a: bench.ModelBundle,
    bundle_b: bench.ModelBundle,
    items: Sequence[PromptItem],
    depth_a: int,
    depth_b: int,
) -> PairActivations:
    import torch

    rows: list[dict[str, Any]] = []
    vecs_a = []
    vecs_b = []
    for i, item in enumerate(items):
        va = capture_vector(bundle_a, item.text, depth_a)
        vb = capture_vector(bundle_b, item.text, depth_b)
        vecs_a.append(va)
        vecs_b.append(vb)
        rows.append({
            "prompt_id": item.prompt_id,
            "family": item.family,
            "source": item.source,
            "variant": item.variant,
            "text_sha256": hashlib.sha256(item.text.encode("utf-8")).hexdigest(),
            "text_excerpt": item.text[:180].replace("\n", "\\n"),
            "tokens_model_a": tokenize_len(bundle_a, item.text),
            "tokens_model_b": tokenize_len(bundle_b, item.text),
            "norm_model_a": rounded(float(va.norm())),
            "norm_model_b": rounded(float(vb.norm())),
        })
        if (i + 1) % max(1, len(items) // 4) == 0:
            print(f"[lab19] collected activations for {i + 1}/{len(items)} prompts")

    return PairActivations(rows, torch.stack(vecs_a), torch.stack(vecs_b), depth_a, depth_b)


class PairedCrosscoder:
    def __init__(self, d_a: int, d_b: int, n_features: int, seed: int):
        import torch

        gen = torch.Generator().manual_seed(int(seed))
        scale_a = 1.0 / math.sqrt(max(1, d_a))
        scale_b = 1.0 / math.sqrt(max(1, d_b))
        self.W_a = torch.randn(d_a, n_features, generator=gen) * scale_a
        self.W_b = torch.randn(d_b, n_features, generator=gen) * scale_b
        self.b = torch.zeros(n_features)
        self.D_a = torch.randn(n_features, d_a, generator=gen) * scale_a
        self.D_b = torch.randn(n_features, d_b, generator=gen) * scale_b

    def parameters(self) -> list[Any]:
        return [self.W_a, self.W_b, self.b, self.D_a, self.D_b]

    def to(self, device: Any) -> "PairedCrosscoder":
        for name in ("W_a", "W_b", "b", "D_a", "D_b"):
            tensor = getattr(self, name)
            setattr(self, name, tensor.to(device).requires_grad_(True))
        return self

    def encode_a(self, x: Any) -> Any:
        import torch

        return torch.relu(x @ self.W_a + self.b)

    def encode_b(self, x: Any) -> Any:
        import torch

        return torch.relu(x @ self.W_b + self.b)

    def reconstruct_a(self, x: Any) -> tuple[Any, Any]:
        z = self.encode_a(x)
        return z @ self.D_a, z

    def reconstruct_b(self, x: Any) -> tuple[Any, Any]:
        z = self.encode_b(x)
        return z @ self.D_b, z


def normalize(x: Any) -> tuple[Any, Any, Any]:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-4)
    return (x - mean) / std, mean, std


def fvu(x: Any, recon: Any) -> float:
    num = float(((x - recon) ** 2).sum())
    denom = float(((x - x.mean(dim=0, keepdim=True)) ** 2).sum())
    return num / max(denom, 1e-9)


def train_crosscoder(ctx: bench.RunContext, acts: PairActivations, seed: int) -> tuple[PairedCrosscoder, dict[str, Any], dict[str, Any]]:
    import torch

    xa, mean_a, std_a = normalize(acts.x_a.float())
    xb, mean_b, std_b = normalize(acts.x_b.float())
    n_features = min(CROSSCODER_FEATURES, max(8, 4 * xa.shape[0]))
    steps = TRAIN_STEPS_TIER_A if ctx.args.tier == "a" else TRAIN_STEPS_TIER_B
    model = PairedCrosscoder(xa.shape[1], xb.shape[1], n_features, seed).to(xa.device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_rows = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        ra, za = model.reconstruct_a(xa)
        rb, zb = model.reconstruct_b(xb)
        recon = ((ra - xa) ** 2).mean() + ((rb - xb) ** 2).mean()
        sparse = za.mean() + zb.mean()
        loss = recon + L1_WEIGHT * sparse
        loss.backward()
        opt.step()
        if step == 0 or (step + 1) % max(1, steps // 8) == 0:
            loss_rows.append({
                "step": step + 1,
                "loss": rounded(float(loss.detach())),
                "reconstruction_loss": rounded(float(recon.detach())),
                "l1_activation_mean": rounded(float(sparse.detach())),
            })

    with torch.no_grad():
        ra, za = model.reconstruct_a(xa)
        rb, zb = model.reconstruct_b(xb)
    metrics = {
        "n_features": n_features,
        "train_steps": steps,
        "learning_rate": LEARNING_RATE,
        "l1_weight": L1_WEIGHT,
        "fvu_model_a": rounded(fvu(xa, ra)),
        "fvu_model_b": rounded(fvu(xb, rb)),
        "mean_l0_model_a": rounded(float((za > 1e-6).float().sum(dim=1).mean())),
        "mean_l0_model_b": rounded(float((zb > 1e-6).float().sum(dim=1).mean())),
    }
    stats = {
        "mean_a": mean_a,
        "std_a": std_a,
        "mean_b": mean_b,
        "std_b": std_b,
        "z_a": za.detach().cpu(),
        "z_b": zb.detach().cpu(),
    }
    loss_path = ctx.path("tables", "crosscoder_training_curve.csv")
    bench.write_csv_with_context(ctx, loss_path, loss_rows)
    ctx.register_artifact(loss_path, "table", "Crosscoder training loss snapshots.")
    return model, metrics, stats


def classify_feature(base_mean: float, compare_mean: float, corr: float) -> str:
    total = base_mean + compare_mean
    if total < 1e-5:
        return "dead"
    ratio = compare_mean / total
    if corr >= 0.55 and 0.35 <= ratio <= 0.65:
        return "shared"
    if ratio >= 0.72:
        return "instruct_skewed"
    if ratio <= 0.28:
        return "base_skewed"
    return "asymmetric"


def feature_taxonomy_rows(model: PairedCrosscoder, stats: Mapping[str, Any]) -> list[dict[str, Any]]:
    import torch

    z_a = stats["z_a"]
    z_b = stats["z_b"]
    rows = []
    dec_norm_a = torch.linalg.vector_norm(model.D_a.detach().cpu(), dim=1)
    dec_norm_b = torch.linalg.vector_norm(model.D_b.detach().cpu(), dim=1)
    for fid in range(z_a.shape[1]):
        a_vals = z_a[:, fid].tolist()
        b_vals = z_b[:, fid].tolist()
        a_mean = safe_fmean(a_vals, 0.0)
        b_mean = safe_fmean(b_vals, 0.0)
        corr = pearson(a_vals, b_vals)
        total = a_mean + b_mean
        rows.append({
            "feature_id": fid,
            "taxonomy": classify_feature(a_mean, b_mean, corr),
            "activation_mean_model_a": rounded(a_mean),
            "activation_mean_model_b": rounded(b_mean),
            "compare_activation_share": rounded(b_mean / total if total > 1e-9 else float("nan")),
            "activation_correlation": none_if_nan(corr),
            "decoder_norm_model_a": rounded(float(dec_norm_a[fid])),
            "decoder_norm_model_b": rounded(float(dec_norm_b[fid])),
            "decoder_norm_compare_share": rounded(float(dec_norm_b[fid] / (dec_norm_a[fid] + dec_norm_b[fid] + 1e-9))),
        })
    return rows


def gallery_rows(
    taxonomy: Sequence[Mapping[str, Any]],
    stats: Mapping[str, Any],
    prompts: Sequence[PromptItem],
) -> list[dict[str, Any]]:
    z_a = stats["z_a"]
    z_b = stats["z_b"]
    candidates = [
        row for row in taxonomy
        if row["taxonomy"] in {"instruct_skewed", "base_skewed", "shared", "asymmetric"}
    ][:GALLERY_FEATURES]
    rows = []
    for row in candidates:
        fid = int(row["feature_id"])
        if row["taxonomy"] == "base_skewed":
            scores = z_a[:, fid]
        elif row["taxonomy"] == "instruct_skewed":
            scores = z_b[:, fid]
        else:
            scores = z_a[:, fid] + z_b[:, fid]
        top = sorted(range(len(prompts)), key=lambda i: float(scores[i]), reverse=True)[:GALLERY_CONTEXTS]
        for rank, idx in enumerate(top, start=1):
            prompt = prompts[idx]
            rows.append({
                "feature_id": fid,
                "taxonomy": row["taxonomy"],
                "rank": rank,
                "score_model_a": rounded(float(z_a[idx, fid])),
                "score_model_b": rounded(float(z_b[idx, fid])),
                "family": prompt.family,
                "variant": prompt.variant,
                "source": prompt.source,
                "prompt_id": prompt.prompt_id,
                "text_excerpt": prompt.text[:220].replace("\n", "\\n"),
                "proposed_label": "",
                "label_status": "student_label_required",
            })
    return rows


def voice_marker_rows(prompts: Sequence[PromptItem]) -> list[dict[str, Any]]:
    rows = []
    for family in sorted({p.family for p in prompts}):
        for variant in sorted({p.variant for p in prompts if p.family == family}):
            sub = [p for p in prompts if p.family == family and p.variant == variant]
            marker_rates = []
            for prompt in sub:
                low = prompt.text.lower()
                marker_rates.append(1.0 if any(marker in low for marker in VOICE_MARKERS) else 0.0)
            rows.append({
                "family": family,
                "variant": variant,
                "n_prompts": len(sub),
                "assistant_voice_marker_rate_in_prompt_text": rounded(safe_fmean(marker_rates)),
                "note": "Prompt-text marker control only; generation markers need --run-edit or a separate behavioral run.",
            })
    return rows


def direction_bridge_rows(model: PairedCrosscoder, bundle_b: bench.ModelBundle) -> list[dict[str, Any]]:
    rows = []
    state_candidates = [
        ("lab17_persona", bench.COURSE_ROOT / "runs"),
    ]
    # This starter does not guess which run directory contains the student's
    # saved directions. It writes a clear placeholder so a Colab notebook can
    # join the chosen state file explicitly.
    for name, root in state_candidates:
        rows.append({
            "bridge": name,
            "status": "not_auto_resolved",
            "expected_d_model": bundle_b.anatomy.d_model,
            "feature_decoder_space": "comparison_model_normalized_residual_space",
            "note": f"Load a chosen state/*.pt file from {root} and compute decoder-direction cosines in the notebook extension.",
        })
    return rows


def plot_exclusivity(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(8.4, 5.0))
    vals = [
        float(row["compare_activation_share"])
        for row in rows
        if isinstance(row.get("compare_activation_share"), (int, float))
    ]
    if vals:
        ax.hist(vals, bins=16, color="#326b7c", alpha=0.85)
    ax.set_xlabel("comparison-model activation share")
    ax.set_ylabel("feature count")
    ax.set_title("Crosscoder feature exclusivity")
    bench.style_ax(ax, legend=False)
    bench.save_figure(ctx, fig, "feature_exclusivity_histogram.png", "Histogram of model-skew scores across crosscoder features.")


def plot_direction_bridge_placeholder(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(7.2, 3.6))
    ax.axis("off")
    ax.text(0.02, 0.72, "Direction bridge is a notebook extension in this starter run.", fontsize=11)
    ax.text(0.02, 0.48, "Join a chosen Lab 16/17/18 state file to feature decoders.", fontsize=10)
    ax.text(0.02, 0.24, f"Rows emitted: {len(rows)}", fontsize=10)
    bench.save_figure(ctx, fig, "feature_direction_bridge.png", "Placeholder for feature-direction bridge extension.")


def run_optional_causal_validation(
    ctx: bench.RunContext,
    bundle_b: bench.ModelBundle,
    model: PairedCrosscoder,
    taxonomy: Sequence[Mapping[str, Any]],
    stats: Mapping[str, Any],
    depth_b: int,
) -> list[dict[str, Any]]:
    import torch

    candidates = [row for row in taxonomy if row["taxonomy"] == "instruct_skewed"]
    if not candidates:
        return [{
            "status": "skipped_no_instruct_skewed_feature",
            "feature_id": "",
            "condition": "",
            "marker_rate": "",
            "generation": "",
        }]
    feature_id = int(max(candidates, key=lambda r: float(r.get("compare_activation_share") or 0.0))["feature_id"])
    # Decoder lives in normalized comparison activation space; map it back to
    # residual units before using it as an activation-addition vector.
    vec = model.D_b.detach().cpu()[feature_id] * stats["std_b"].detach().cpu()[0]
    vec = vec / vec.norm().clamp_min(1e-9)
    layer = max(0, min(bundle_b.anatomy.n_layers - 1, depth_b - 1))
    prompts = [
        render_for_compare_chat(bundle_b, "Please answer in your default assistant style: how should I organize project notes?"),
        render_for_compare_chat(bundle_b, "Please answer carefully: what should I do if I am unsure about a factual claim?"),
        render_for_compare_chat(bundle_b, "Please answer briefly: how do I write a polite correction?"),
    ]
    rows = []
    random_vec = torch.randn_like(vec)
    random_vec = random_vec / random_vec.norm().clamp_min(1e-9)
    for condition, steer in (
        ("baseline", None),
        ("feature_plus", (layer, vec, 0.7)),
        ("feature_minus", (layer, vec, -0.7)),
        ("random_plus", (layer, random_vec, 0.7)),
    ):
        outs = bench.generate_continuous(
            bundle_b,
            prompts,
            MAX_NEW_TOKENS,
            max_concurrent=ENGINE_MAX_CONCURRENT,
            progress_label=f"lab19 causal {condition}",
            steer=steer,
        )
        for prompt, text in zip(prompts, outs):
            low = text.lower()
            rows.append({
                "status": "ran",
                "feature_id": feature_id,
                "condition": condition,
                "layer": layer,
                "marker_rate": 1.0 if any(marker in low for marker in VOICE_MARKERS) else 0.0,
                "generation": text,
                "prompt_excerpt": prompt[-180:].replace("\n", "\\n"),
                "hand_label_behavior": "",
            })
    return rows


def write_report(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 19 Model Diffing Report",
        "",
        f"- Model A: `{metrics.get('model_a')}`",
        f"- Model B: `{metrics.get('model_b')}`",
        f"- Prompts: {metrics.get('n_prompts')}",
        f"- Stream depths: A={metrics.get('depth_a')}, B={metrics.get('depth_b')}",
        f"- Features: {metrics.get('n_features')}",
        f"- FVU A/B: {metrics.get('fvu_model_a')} / {metrics.get('fvu_model_b')}",
        f"- Taxonomy counts: {metrics.get('taxonomy_counts')}",
        "",
        "Read `operationalization_audit.md` before calling a feature a default-assistant-voice feature. The first explanation to try is template distribution or prompt-family imbalance.",
        "",
    ]
    path = ctx.path("model_diffing_report.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Human-readable Lab 19 model-diffing report.")


def write_operationalization_audit(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 19 Operationalization Audit",
        "",
        "## What Was Measured",
        "",
        "The lab trains a small paired crosscoder over matched final-token residual activations from two models on the same prompt inventory.",
        "",
        "It does not prove that instruction following, alignment, personality, or assistant identity is localized in a feature.",
        "",
        "## Cheap Explanations",
        "",
        "- Template tokens: compare `variant=raw` and `variant=compare_chat` rows before naming a feature.",
        "- Distribution mismatch: inspect `prompt_inventory.csv`; a model-skewed feature may just be a family-skewed prompt subset.",
        "- Norm shifts: check `activation_norms.csv` and FVU before reading feature labels.",
        "- Crosscoder artifact: random or dead features can look exclusive when the dictionary is too small.",
        "- Marker habits: `default_voice_marker_rates.csv` is a prompt-text control, not behavioral proof.",
        "",
        "## Current Run",
        "",
        f"- FVU model A: {metrics.get('fvu_model_a')}",
        f"- FVU model B: {metrics.get('fvu_model_b')}",
        f"- Shared features: {metrics.get('taxonomy_counts', {}).get('shared')}",
        f"- Base-skewed features: {metrics.get('taxonomy_counts', {}).get('base_skewed')}",
        f"- Instruct-skewed features: {metrics.get('taxonomy_counts', {}).get('instruct_skewed')}",
        "",
        "## Allowed Claim",
        "",
        "A feature-level model-diff claim is allowed only after feature exclusivity survives matched-prompt and template controls. Mechanism language needs the optional causal validation or a follow-up patch/clamp.",
        "",
    ]
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "audit", "Operationalization limits and cheap-explanation audit for Lab 19.")


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    import torch

    args = ctx.args
    compare_id, compare_revision = comparison_model_id(ctx)
    compare_bundle = load_comparison_bundle(ctx, compare_id, compare_revision)
    depth_a, depth_b = selected_depths(ctx, bundle, compare_bundle)

    prompts, prompt_info = load_prompt_inventory(args, compare_bundle)
    prompt_path = ctx.path("tables", "prompt_inventory.csv")
    bench.write_csv_with_context(ctx, prompt_path, [dataclasses.asdict(p) for p in prompts])
    ctx.register_artifact(prompt_path, "table", "Prompt inventory for Lab 19 matched activation collection.")

    pair_info = {
        "model_a": bundle.anatomy.model_id,
        "model_b": compare_bundle.anatomy.model_id,
        "model_a_revision": bundle.anatomy.revision,
        "model_b_revision": compare_bundle.anatomy.revision,
        "depth_a": depth_a,
        "depth_b": depth_b,
        "d_model_a": bundle.anatomy.d_model,
        "d_model_b": compare_bundle.anatomy.d_model,
        "prompt_inventory": prompt_info,
        "tier_a_note": "The default Tier A pair is an identity-pair smoke test; use Tier B or LAB19_COMPARE_MODEL for science.",
    }
    pair_path = ctx.path("diagnostics", "model_pair.json")
    bench.write_json(pair_path, pair_info)
    ctx.register_artifact(pair_path, "diagnostic", "Model-pair metadata for Lab 19.")

    acts = collect_pair_activations(ctx, bundle, compare_bundle, prompts, depth_a, depth_b)
    act_path = ctx.path("diagnostics", "activation_norms.csv")
    bench.write_csv_with_context(ctx, act_path, acts.prompt_rows)
    ctx.register_artifact(act_path, "diagnostic", "Prompt-level token counts and residual norm controls for the model pair.")

    crosscoder, train_metrics, stats = train_crosscoder(ctx, acts, int(args.seed))
    taxonomy = feature_taxonomy_rows(crosscoder, stats)
    taxonomy_path = ctx.path("tables", "feature_taxonomy.csv")
    bench.write_csv_with_context(ctx, taxonomy_path, taxonomy)
    ctx.register_artifact(taxonomy_path, "table", "Crosscoder feature taxonomy with shared/base-skew/instruct-skew labels.")
    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, taxonomy)
    ctx.register_artifact(results_path, "results", "Alias of feature_taxonomy.csv for the standard run contract.")

    gallery = gallery_rows(taxonomy, stats, prompts)
    gallery_path = ctx.path("tables", "instruct_only_feature_gallery.csv")
    bench.write_csv_with_context(ctx, gallery_path, gallery)
    ctx.register_artifact(gallery_path, "table", "Top prompt contexts for selected crosscoder features.")

    marker_path = ctx.path("tables", "default_voice_marker_rates.csv")
    bench.write_csv_with_context(ctx, marker_path, voice_marker_rows(prompts))
    ctx.register_artifact(marker_path, "table", "Prompt-text default-assistant marker control rates by family and variant.")

    bridge = direction_bridge_rows(crosscoder, compare_bundle)
    bridge_path = ctx.path("tables", "feature_direction_bridge.csv")
    bench.write_csv_with_context(ctx, bridge_path, bridge)
    ctx.register_artifact(bridge_path, "table", "Placeholder bridge table for joining crosscoder features to saved direction state files.")

    if args.run_edit:
        causal_rows = run_optional_causal_validation(ctx, compare_bundle, crosscoder, taxonomy, stats, depth_b)
    else:
        causal_rows = [{
            "status": "skipped",
            "feature_id": "",
            "condition": "",
            "marker_rate": "",
            "generation": "",
            "note": "Rerun Lab 19 with --run-edit to perform the optional feature-intervention smoke test.",
        }]
    causal_path = ctx.path("tables", "causal_feature_validation.csv")
    bench.write_csv_with_context(ctx, causal_path, causal_rows)
    ctx.register_artifact(causal_path, "table", "Optional causal feature validation; skipped unless --run-edit is set.")

    state = {
        "model_a": bundle.anatomy.model_id,
        "model_b": compare_bundle.anatomy.model_id,
        "depth_a": depth_a,
        "depth_b": depth_b,
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
    }
    state_path = ctx.path("state", "crosscoder_state.pt")
    torch.save(state, state_path)
    ctx.register_artifact(state_path, "tensor", "Trained paired crosscoder weights, normalization, and taxonomy.")

    taxonomy_counts = dict(Counter(row["taxonomy"] for row in taxonomy))
    metrics = {
        **pair_info,
        **train_metrics,
        "n_prompts": len(prompts),
        "taxonomy_counts": taxonomy_counts,
        "n_gallery_rows": len(gallery),
        "causal_validation_status": dict(Counter(row.get("status", "") for row in causal_rows)),
    }
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 19 metrics.")

    if not args.no_plots:
        plot_exclusivity(ctx, taxonomy)
        plot_direction_bridge_placeholder(ctx, bridge)

    write_report(ctx, metrics)
    write_operationalization_audit(ctx, metrics)

    run_name = ctx.run_dir.name
    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "DECODE/ATTR",
            "text": (
                f"At residual depths A={depth_a}, B={depth_b}, the paired crosscoder found "
                f"{taxonomy_counts.get('instruct_skewed', 0)} comparison-skewed features and "
                f"{taxonomy_counts.get('base_skewed', 0)} base-skewed features with FVU "
                f"{metrics['fvu_model_a']} / {metrics['fvu_model_b']}. This is a model-pair feature taxonomy under the sampled prompt distribution."
            ),
            "artifact": f"runs/{run_name}/tables/feature_taxonomy.csv",
            "falsifier": "Template controls, matched prompt families, or a larger dictionary erase the exclusivity pattern.",
        },
        {
            "id": f"{LAB_ID}-C2",
            "tag": "CAUSAL",
            "text": (
                "Optional feature-intervention evidence is present only if `tables/causal_feature_validation.csv` has status `ran`; otherwise Lab 19 has not made a causal feature claim."
            ),
            "artifact": f"runs/{run_name}/tables/causal_feature_validation.csv",
            "falsifier": "Random-feature intervention matches the effect or hand labels reject the marker-based behavior score.",
        },
    ]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
