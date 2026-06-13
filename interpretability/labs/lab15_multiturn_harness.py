"""Lab 15: Multi-turn instrumentation.

This is a harness-validation lab, not a model-science lab. Later labs want
to say things like "the persona state rose over turns" or "the model changed
its belief after pushback." Those claims are worthless if the turn boundary,
chat template, KV-cache, or patching conventions are off by one token.

The demo conversation is harmless and scripted. We trace a topic direction
through an accumulating orchid-greenhouse conversation and compare it with
random and length-matched null directions. The headline artifact is not the
topic trace itself; it is the stack of self-checks that says turn-indexed
measurements are well-defined under this tokenizer/model pair.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from typing import Any, Mapping, Sequence

import interp_bench as bench

LAB_ID = "L15"

SYSTEM_PROMPT = (
    "You are a precise assistant. Keep replies short and preserve the user's "
    "topic labels exactly."
)

CACHE_PARITY_ATOL = 2e-3
PATCH_NOOP_ATOL = 2e-3
TRACE_DEPTH_FRACTION = 0.5


@dataclasses.dataclass
class ConversationSpec:
    name: str
    messages: list[dict[str, str]]
    expected_topic: str


@dataclasses.dataclass
class Segment:
    index: int
    role: str
    start: int
    end: int
    content: str


def rounded(x: Any, ndigits: int = 6) -> Any:
    try:
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            return round(float(x), ndigits)
    except Exception:
        pass
    return x


def safe_fmean(vals: Sequence[float], default: float = float("nan")) -> float:
    finite = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(statistics.fmean(finite)) if finite else default


def slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return float("nan")
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    mx = statistics.fmean(xvals)
    my = statistics.fmean(yvals)
    denom = sum((x - mx) ** 2 for x in xvals)
    if denom < 1e-12:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in pairs) / denom


def unit(v: Any) -> Any:
    norm = v.norm().clamp_min(1e-9)
    if not bool(norm.isfinite()):
        raise RuntimeError("Direction norm was not finite.")
    return v / norm


def random_unit(d_model: int, seed: int) -> Any:
    import torch

    gen = torch.Generator().manual_seed(seed)
    return unit(torch.randn(d_model, generator=gen))


def conversations() -> list[ConversationSpec]:
    topic_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "We are planning a small orchid greenhouse. Remember that orchids are the project focus."},
        {"role": "assistant", "content": "Noted: the greenhouse plan is centered on orchids."},
        {"role": "user", "content": "Add humidity, bark medium, and filtered light as orchid-care requirements."},
        {"role": "assistant", "content": "The orchid plan now includes humidity, airy bark medium, and filtered light."},
        {"role": "user", "content": "Add watering rhythm and airflow for the orchid benches."},
        {"role": "assistant", "content": "The orchid benches should balance watering rhythm, airflow, and spacing."},
        {"role": "user", "content": "Summarize the greenhouse plan with the orchid priorities in order."},
        {"role": "assistant", "content": "Orchid priorities: humidity, filtered light, bark medium, airflow, and careful watering."},
    ]
    control_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "We are planning a small archive room. Remember that folders are the project focus."},
        {"role": "assistant", "content": "Noted: the archive plan is centered on folders."},
        {"role": "user", "content": "Add labels, shelf spacing, and climate notes as folder-care requirements."},
        {"role": "assistant", "content": "The folder plan now includes labels, shelf spacing, and climate notes."},
        {"role": "user", "content": "Add review rhythm and aisle airflow for the folder shelves."},
        {"role": "assistant", "content": "The folder shelves should balance review rhythm, airflow, and spacing."},
        {"role": "user", "content": "Summarize the archive plan with the folder priorities in order."},
        {"role": "assistant", "content": "Folder priorities: labels, shelf spacing, climate notes, airflow, and careful review."},
    ]
    return [
        ConversationSpec("orchid_topic", topic_messages, "orchid"),
        ConversationSpec("archive_length_control", control_messages, "folder"),
    ]


def render_messages(bundle: bench.ModelBundle, messages: Sequence[Mapping[str, str]]) -> str:
    return bundle.tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=False
    )


def token_ids(bundle: bench.ModelBundle, rendered: str) -> list[int]:
    return list(bundle.tokenizer(rendered, add_special_tokens=False)["input_ids"])


def direct_template_ids(bundle: bench.ModelBundle, messages: Sequence[Mapping[str, str]]) -> list[int]:
    ids = bundle.tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=False
    )
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def build_segments(bundle: bench.ModelBundle, conv: ConversationSpec) -> tuple[str, list[int], list[Segment], dict[str, Any]]:
    rendered = render_messages(bundle, conv.messages)
    full_ids = token_ids(bundle, rendered)
    direct_ids = direct_template_ids(bundle, conv.messages)
    segments: list[Segment] = []
    stable_prefix = True
    prev_ids: list[int] = []
    for idx, message in enumerate(conv.messages):
        partial_ids = token_ids(bundle, render_messages(bundle, conv.messages[: idx + 1]))
        if partial_ids[: len(prev_ids)] != prev_ids:
            stable_prefix = False
        segments.append(
            Segment(
                index=idx,
                role=message["role"],
                start=len(prev_ids),
                end=len(partial_ids),
                content=message["content"],
            )
        )
        prev_ids = partial_ids
    info = {
        "conversation": conv.name,
        "rendered_token_count": len(full_ids),
        "direct_token_count": len(direct_ids),
        "string_vs_direct_template_ids_match": full_ids == direct_ids,
        "incremental_prefix_stable": stable_prefix,
        "final_incremental_ids_match": prev_ids == full_ids,
    }
    return rendered, full_ids, segments, info


def write_turn_boundary_check(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    specs: Sequence[ConversationSpec],
) -> tuple[dict[str, tuple[str, list[int], list[Segment]]], dict[str, Any]]:
    by_name: dict[str, tuple[str, list[int], list[Segment]]] = {}
    records: list[dict[str, Any]] = []
    all_ok = True
    for conv in specs:
        rendered, ids, segments, info = build_segments(bundle, conv)
        by_name[conv.name] = (rendered, ids, segments)
        coverage_ok = bool(segments) and segments[0].start == 0 and segments[-1].end == len(ids)
        no_gaps = all(a.end == b.start for a, b in zip(segments, segments[1:]))
        positive_widths = all(seg.end > seg.start for seg in segments)
        no_assistant_leak = True
        leaks: list[str] = []
        for i, seg in enumerate(segments):
            if seg.role != "user":
                continue
            next_assistant = next((s for s in segments[i + 1:] if s.role == "assistant"), None)
            if next_assistant is None:
                continue
            decoded = bundle.tokenizer.decode(ids[seg.start: seg.end])
            snippet = next_assistant.content[: min(24, len(next_assistant.content))]
            if snippet and snippet in decoded:
                no_assistant_leak = False
                leaks.append(f"user segment {seg.index} contains assistant snippet {snippet!r}")
        ok = (
            coverage_ok
            and no_gaps
            and positive_widths
            and no_assistant_leak
            and info["string_vs_direct_template_ids_match"]
            and info["incremental_prefix_stable"]
            and info["final_incremental_ids_match"]
        )
        all_ok = all_ok and ok
        records.append({
            **info,
            "coverage_ok": coverage_ok,
            "no_gaps": no_gaps,
            "positive_widths": positive_widths,
            "no_assistant_text_in_user_spans": no_assistant_leak,
            "leaks": leaks,
            "ok": ok,
            "segments": [
                {
                    "index": seg.index,
                    "role": seg.role,
                    "start": seg.start,
                    "end": seg.end,
                    "n_tokens": seg.end - seg.start,
                    "content_excerpt": seg.content[:80],
                }
                for seg in segments
            ],
        })
    result = {
        "ok": all_ok,
        "conversations": records,
        "explanation": (
            "Each span is derived by repeatedly rendering prefixes with the model's own "
            "chat template. The checks require stable prefix tokenization, complete "
            "coverage, no gaps, and no assistant content inside user spans."
        ),
    }
    path = ctx.path("diagnostics", "turn_boundary_check.json")
    bench.write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "Chat-template turn segmentation and template parity checks.")
    if not all_ok:
        raise RuntimeError("Turn-boundary/template parity failed; see diagnostics/turn_boundary_check.json.")
    return by_name, result


def run_ids(
    bundle: bench.ModelBundle,
    ids: Sequence[int],
    *,
    past_key_values: Any = None,
    use_cache: bool = False,
) -> tuple[Any, Any, Any]:
    import torch

    input_ids = torch.tensor([list(ids)], device=bundle.input_device)
    captured: dict[str, Any] = {}

    def final_norm_pre_hook(module: Any, hook_args: tuple) -> None:
        captured["final_prenorm"] = bench.tensor_cpu_float(hook_args[0])

    handle = bundle.final_norm.register_forward_pre_hook(final_norm_pre_hook)
    try:
        with torch.no_grad():
            out = bundle.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=use_cache,
            )
    finally:
        handle.remove()
    if "final_prenorm" not in captured:
        raise RuntimeError("Final-norm pre-hook did not fire during multi-turn capture.")
    streams = torch.stack(
        [bench.tensor_cpu_float(h[0]) for h in out.hidden_states[:-1]]
        + [captured["final_prenorm"][0]]
    )
    return streams, bench.tensor_cpu_float(out.logits[0, -1]), getattr(out, "past_key_values", None)


def full_recompute_boundary_streams(bundle: bench.ModelBundle, ids: Sequence[int], boundaries: Sequence[int]) -> list[Any]:
    out = []
    for boundary in boundaries:
        streams, _, _ = run_ids(bundle, ids[: boundary + 1], use_cache=False)
        out.append(streams[:, -1, :])
    return out


def incremental_boundary_streams(bundle: bench.ModelBundle, ids: Sequence[int], boundaries: Sequence[int]) -> list[Any]:
    out = []
    past = None
    prev = 0
    for boundary in boundaries:
        chunk = ids[prev: boundary + 1]
        if not chunk:
            raise RuntimeError("Empty incremental chunk while computing KV-cache parity.")
        streams, _, past = run_ids(bundle, chunk, past_key_values=past, use_cache=True)
        out.append(streams[:, -1, :])
        prev = boundary + 1
    return out


def write_cache_parity_check(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    conv_name: str,
    ids: Sequence[int],
    segments: Sequence[Segment],
) -> dict[str, Any]:
    import torch

    boundaries = [seg.end - 1 for seg in segments]
    full = full_recompute_boundary_streams(bundle, ids, boundaries)
    cached = incremental_boundary_streams(bundle, ids, boundaries)
    rows = []
    worst = 0.0
    worst_mean = 0.0
    for seg, a, b in zip(segments, full, cached):
        diff = (a - b).abs()
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        worst = max(worst, max_diff)
        worst_mean = max(worst_mean, mean_diff)
        rows.append({
            "conversation": conv_name,
            "segment_index": seg.index,
            "role": seg.role,
            "boundary_token": seg.end - 1,
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
            "ok_at_tolerance": max_diff <= CACHE_PARITY_ATOL,
        })
    csv_path = ctx.path("diagnostics", "cache_recompute_parity_by_boundary.csv")
    bench.write_csv_with_context(ctx, csv_path, rows)
    ctx.register_artifact(csv_path, "diagnostic", "Boundary-level KV-cache capture versus full-recompute parity.")
    result = {
        "conversation": conv_name,
        "n_boundaries": len(boundaries),
        "max_abs_diff": worst,
        "max_mean_abs_diff": worst_mean,
        "atol": CACHE_PARITY_ATOL,
        "ok": worst <= CACHE_PARITY_ATOL,
        "explanation": (
            "Each conversation prefix was measured two ways: full recompute of the prefix "
            "and incremental prefill with past_key_values. Boundary residuals must match "
            "or later turn-indexed traces may be cache artifacts."
        ),
    }
    path = ctx.path("diagnostics", "cache_recompute_parity.json")
    bench.write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "KV-cache-aware boundary capture parity against full recompute.")
    if not result["ok"]:
        raise RuntimeError("KV-cache parity failed; see diagnostics/cache_recompute_parity.json.")
    # Keep torch imported in this function so py_compile catches its use; no-op.
    _ = torch
    return result


def logits_with_self_patch(
    bundle: bench.ModelBundle,
    ids: Sequence[int],
    *,
    layer: int,
    pos: int,
    vector: Any,
) -> Any:
    import torch

    input_ids = torch.tensor([list(ids)], device=bundle.input_device)
    block = bundle.blocks[layer]

    def patch_hook(module: Any, hook_args: tuple, output: Any) -> Any:
        out = output[0] if isinstance(output, tuple) else output
        patched = out.clone()
        patched[:, pos, :] = vector.to(patched.device, patched.dtype)
        if isinstance(output, tuple):
            return (patched,) + tuple(output[1:])
        return patched

    handle = block.register_forward_hook(patch_hook)
    try:
        with torch.no_grad():
            out = bundle.model(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()
    return bench.tensor_cpu_float(out.logits[0, -1])


def write_patch_noop_check(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    conv_name: str,
    ids: Sequence[int],
    segments: Sequence[Segment],
) -> dict[str, Any]:
    n_layers = bundle.anatomy.n_layers
    layer = min(n_layers - 1, max(0, n_layers // 2))
    depth = layer + 1
    user_boundaries = [seg.end - 1 for seg in segments if seg.role == "user"]
    pos = user_boundaries[-1] if user_boundaries else segments[-1].end - 1
    streams, base_logits, _ = run_ids(bundle, ids, use_cache=False)
    patched_logits = logits_with_self_patch(bundle, ids, layer=layer, pos=pos, vector=streams[depth, pos])
    max_diff = float((patched_logits - base_logits).abs().max())
    mean_diff = float((patched_logits - base_logits).abs().mean())
    result = {
        "conversation": conv_name,
        "layer": layer,
        "stream_depth": depth,
        "position": pos,
        "max_abs_logit_diff": max_diff,
        "mean_abs_logit_diff": mean_diff,
        "atol": PATCH_NOOP_ATOL,
        "ok": max_diff <= PATCH_NOOP_ATOL,
        "explanation": (
            "A decoder block output at a real turn boundary was replaced with the same "
            "run's cached vector. This should be an identity operation; otherwise later "
            "cross-turn patching claims would not target the named stream."
        ),
    }
    path = ctx.path("diagnostics", "patch_noop_check.json")
    bench.write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "Self-patching a turn-boundary stream is a no-op.")
    if not result["ok"]:
        raise RuntimeError("Turn-boundary patch no-op failed; see diagnostics/patch_noop_check.json.")
    return result


def direction_prompt(topic: str) -> list[dict[str, str]]:
    if topic == "orchid":
        user = "Topic focus: orchid greenhouse humidity bark medium filtered light watering airflow."
        assistant = "The notes are about orchid care in a greenhouse."
    elif topic == "archive":
        user = "Topic focus: archive room labels shelf spacing climate notes review airflow."
        assistant = "The notes are about folder care in an archive room."
    elif topic == "neutral_a":
        user = "Topic focus: schedule notes columns rows markers entries counts."
        assistant = "The notes are about neutral planning fields."
    elif topic == "neutral_b":
        user = "Topic focus: ledger items rows columns markers entries counts."
        assistant = "The notes are about neutral planning records."
    else:
        raise ValueError(topic)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def final_vec_for_messages(bundle: bench.ModelBundle, messages: Sequence[Mapping[str, str]], depth: int) -> Any:
    ids = token_ids(bundle, render_messages(bundle, messages))
    streams, _, _ = run_ids(bundle, ids, use_cache=False)
    return streams[depth, -1, :]


def build_trace_directions(bundle: bench.ModelBundle, depth: int, seed: int) -> dict[str, Any]:
    import torch

    topic_pairs = []
    for _ in range(3):
        topic_pairs.append(
            final_vec_for_messages(bundle, direction_prompt("orchid"), depth)
            - final_vec_for_messages(bundle, direction_prompt("archive"), depth)
        )
    length_pairs = []
    for _ in range(3):
        length_pairs.append(
            final_vec_for_messages(bundle, direction_prompt("neutral_a"), depth)
            - final_vec_for_messages(bundle, direction_prompt("neutral_b"), depth)
        )
    return {
        "topic_orchid_minus_archive": unit(torch.stack(topic_pairs).mean(dim=0)),
        "length_matched_null": unit(torch.stack(length_pairs).mean(dim=0)),
        "random_null": random_unit(bundle.anatomy.d_model, seed),
    }


def projection_rows(
    bundle: bench.ModelBundle,
    by_name: Mapping[str, tuple[str, list[int], list[Segment]]],
    directions: Mapping[str, Any],
    depth: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for conv_name, (_, ids, segments) in by_name.items():
        streams, _, _ = run_ids(bundle, ids, use_cache=False)
        non_system_seen = 0
        for seg in segments:
            if seg.role != "system":
                non_system_seen += 1
            span = streams[depth, seg.start: seg.end, :]
            span_mean = span.mean(dim=0)
            boundary = streams[depth, seg.end - 1, :]
            cumulative = streams[depth, : seg.end, :].mean(dim=0)
            for direction_name, direction in directions.items():
                rows.append({
                    "conversation": conv_name,
                    "segment_index": seg.index,
                    "turn_index_non_system": non_system_seen,
                    "role": seg.role,
                    "start_token": seg.start,
                    "end_token_exclusive": seg.end,
                    "n_tokens": seg.end - seg.start,
                    "direction": direction_name,
                    "span_mean_projection": rounded(float(span_mean @ direction)),
                    "boundary_projection": rounded(float(boundary @ direction)),
                    "cumulative_projection": rounded(float(cumulative @ direction)),
                    "content_excerpt": seg.content[:80],
                })
    return rows


def write_null_trace_check(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    slope_rows: list[dict[str, Any]] = []
    for conv in sorted({r["conversation"] for r in rows}):
        for direction in sorted({r["direction"] for r in rows}):
            sub = [
                r for r in rows
                if r["conversation"] == conv and r["direction"] == direction and r["role"] != "system"
            ]
            xs = [float(r["turn_index_non_system"]) for r in sub]
            ys = [float(r["cumulative_projection"]) for r in sub]
            slope_rows.append({
                "conversation": conv,
                "direction": direction,
                "cumulative_projection_slope": rounded(slope(xs, ys)),
                "start_projection": rounded(ys[0] if ys else float("nan")),
                "end_projection": rounded(ys[-1] if ys else float("nan")),
            })
    csv_path = ctx.path("diagnostics", "null_trace_slopes.csv")
    bench.write_csv_with_context(ctx, csv_path, slope_rows)
    ctx.register_artifact(csv_path, "diagnostic", "Projection slopes for topic and null directions.")
    finite = [
        float(r["cumulative_projection_slope"])
        for r in slope_rows
        if isinstance(r["cumulative_projection_slope"], (int, float))
        and math.isfinite(float(r["cumulative_projection_slope"]))
    ]
    result = {
        "ok": len(finite) == len(slope_rows),
        "slope_rows": slope_rows,
        "explanation": (
            "Null traces are reported as diagnostics rather than scientific claims. "
            "A large monotonic random or length-null slope is a warning that later "
            "multi-turn labs need stronger length/template controls."
        ),
    }
    path = ctx.path("diagnostics", "null_trace_check.json")
    bench.write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "Null projection trace sanity check.")
    if not result["ok"]:
        raise RuntimeError("Null trace slopes were non-finite; see diagnostics/null_trace_check.json.")
    return result


def plot_turn_trace(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(9.2, 5.6))
    styles = {
        ("orchid_topic", "topic_orchid_minus_archive"): ("tab:green", "-", "orchid topic direction"),
        ("orchid_topic", "random_null"): ("tab:gray", ":", "random null"),
        ("orchid_topic", "length_matched_null"): ("tab:orange", ":", "length-matched null"),
        ("archive_length_control", "topic_orchid_minus_archive"): ("tab:blue", "--", "topic direction on control conversation"),
    }
    for (conv, direction), (color, linestyle, label) in styles.items():
        sub = [
            r for r in rows
            if r["conversation"] == conv and r["direction"] == direction and r["role"] != "system"
        ]
        if not sub:
            continue
        ax.plot(
            [float(r["turn_index_non_system"]) for r in sub],
            [float(r["cumulative_projection"]) for r in sub],
            marker="o",
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
    ax.set_xlabel("message boundary (system excluded)")
    ax.set_ylabel("cumulative mean projection")
    ax.set_title("Demo multi-turn trace: topic accumulation vs null controls")
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "demo_turn_trace.png", "Topic and null projection traces over scripted multi-turn conversations.")


def write_report(
    ctx: bench.RunContext,
    metrics: Mapping[str, Any],
    checks: Mapping[str, Mapping[str, Any]],
) -> None:
    lines = [
        "# Lab 15 Multi-Turn Harness Report",
        "",
        "This lab validates the measurement apparatus. The scripted topic trace is a demo, not a claim about model psychology.",
        "",
        "## Self-Checks",
        "",
    ]
    for name, result in checks.items():
        lines.append(f"- `{name}`: {'PASS' if result.get('ok') else 'FAIL'}")
    lines += [
        "",
        "## Headline Diagnostics",
        "",
        f"- Trace depth: {metrics.get('trace_depth')}",
        f"- Cache parity max abs diff: {metrics.get('cache_parity_max_abs_diff')}",
        f"- Patch no-op max abs logit diff: {metrics.get('patch_noop_max_abs_logit_diff')}",
        f"- Topic trace slope: {metrics.get('topic_trace_slope')}",
        f"- Random-null slope: {metrics.get('random_null_slope')}",
        f"- Length-null slope: {metrics.get('length_null_slope')}",
        "",
        "## How Later Labs Should Use This",
        "",
        "Reuse the same pattern: render with the tokenizer's chat template, derive spans from prefix renders, verify cached boundary states against full recompute, and keep a null trace beside every exciting multi-turn projection.",
        "",
    ]
    path = ctx.path("multiturn_harness_report.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Multi-turn instrumentation self-check report.")


def write_operationalization_audit(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 15 Operationalization Audit",
        "",
        "## What Was Measured",
        "",
        "The lab measures whether turn-indexed residual-stream reads are well-defined under the current chat template and model cache behavior. It does not establish a persona, belief, or self-report phenomenon.",
        "",
        "## Cheap Explanations",
        "",
        "- Template residue: checked by string-vs-direct template parity and stable prefix spans.",
        "- Turn leakage: user spans are checked for assistant-text leakage.",
        "- Cache artifacts: incremental KV-cache boundary states are compared against full recompute.",
        "- Patch-hook drift: self-patching a turn-boundary block output must be a no-op.",
        "- Length drift: random and length-matched null directions are traced beside the topic direction.",
        "",
        "## Current Run",
        "",
        f"- Trace depth: {metrics.get('trace_depth')}",
        f"- Cache parity ok: {metrics.get('cache_parity_ok')}",
        f"- Patch no-op ok: {metrics.get('patch_noop_ok')}",
        f"- Null trace finite: {metrics.get('null_trace_ok')}",
        "",
        "## Allowed Claim",
        "",
        "The allowed claim is an instrumentation claim: for this tokenizer/model pair, turn-boundary states can be segmented, cached, recomputed, and self-patched to the reported tolerances. Any later behavioral interpretation must carry its own controls.",
        "",
    ]
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "audit", "Operationalization limits for multi-turn instrumentation.")


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    if not bench.supports_chat_template(bundle):
        raise RuntimeError("Lab 15 requires an instruct/chat model with a chat template.")

    specs = conversations()
    by_name, turn_check = write_turn_boundary_check(ctx, bundle, specs)
    topic_rendered, topic_ids, topic_segments = by_name["orchid_topic"]

    # Standard harness checks on the rendered chat prompt, then the multi-turn
    # specific cache and self-patch checks below.
    bench.run_hook_parity_check(ctx, bundle, topic_rendered)
    bench.run_lens_self_check(ctx, bundle, bench.run_with_residual_cache(bundle, topic_rendered, add_special_tokens=False))

    cache_check = write_cache_parity_check(ctx, bundle, "orchid_topic", topic_ids, topic_segments)
    patch_check = write_patch_noop_check(ctx, bundle, "orchid_topic", topic_ids, topic_segments)

    n_depths = bundle.anatomy.n_layers + 1
    trace_depth = min(n_depths - 1, max(1, int(round(TRACE_DEPTH_FRACTION * bundle.anatomy.n_layers))))
    directions = build_trace_directions(bundle, trace_depth, ctx.args.seed)
    rows = projection_rows(bundle, by_name, directions, trace_depth)
    trace_path = ctx.path("tables", "turn_projection_trace.csv")
    bench.write_csv_with_context(ctx, trace_path, rows)
    ctx.register_artifact(trace_path, "table", "Per-turn topic/random/length-null projections over scripted conversations.")
    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, rows)
    ctx.register_artifact(results_path, "results", "Alias of turn_projection_trace.csv for the standard run contract.")

    null_check = write_null_trace_check(ctx, rows)
    if not ctx.args.no_plots:
        plot_turn_trace(ctx, rows)

    def get_slope(conv: str, direction: str) -> float:
        sub = [
            r for r in rows
            if r["conversation"] == conv and r["direction"] == direction and r["role"] != "system"
        ]
        return slope(
            [float(r["turn_index_non_system"]) for r in sub],
            [float(r["cumulative_projection"]) for r in sub],
        )

    metrics = {
        "model_id": bundle.anatomy.model_id,
        "trace_depth": trace_depth,
        "n_depths": n_depths,
        "n_segments_topic_conversation": len(topic_segments),
        "cache_parity_ok": bool(cache_check["ok"]),
        "cache_parity_max_abs_diff": rounded(cache_check["max_abs_diff"]),
        "patch_noop_ok": bool(patch_check["ok"]),
        "patch_noop_max_abs_logit_diff": rounded(patch_check["max_abs_logit_diff"]),
        "null_trace_ok": bool(null_check["ok"]),
        "topic_trace_slope": rounded(get_slope("orchid_topic", "topic_orchid_minus_archive")),
        "topic_on_control_slope": rounded(get_slope("archive_length_control", "topic_orchid_minus_archive")),
        "random_null_slope": rounded(get_slope("orchid_topic", "random_null")),
        "length_null_slope": rounded(get_slope("orchid_topic", "length_matched_null")),
        "cache_parity_atol": CACHE_PARITY_ATOL,
        "patch_noop_atol": PATCH_NOOP_ATOL,
    }
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 15 instrumentation metrics.")

    checks = {
        "turn_boundary_check": turn_check,
        "cache_recompute_parity": cache_check,
        "patch_noop_check": patch_check,
        "null_trace_check": null_check,
    }
    write_report(ctx, metrics, checks)
    write_operationalization_audit(ctx, metrics)

    run_name = ctx.run_dir.name
    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "OBS",
            "text": (
                f"For {bundle.anatomy.model_id}, the Lab 15 multi-turn harness reproduces "
                f"turn-boundary residual projections under KV-cache prefill versus full recompute "
                f"to max |diff| {metrics['cache_parity_max_abs_diff']} at trace depth {trace_depth}; "
                f"self-patching a boundary state changes logits by max {metrics['patch_noop_max_abs_logit_diff']}. "
                "This is an instrumentation claim, not a model-state interpretation."
            ),
            "artifact": f"runs/{run_name}/diagnostics/cache_recompute_parity.json",
            "falsifier": (
                "Any supported chat template fails segment coverage, cached-vs-recomputed boundary "
                "states exceed tolerance, or self-patching is not a no-op."
            ),
        }
    ]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
