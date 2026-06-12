"""Lab 3: Attention — routing, induction, and what heads actually do.

The lab keeps three different claims in different jars:

1. **Routing (OBS).** Attention patterns say which positions a head reads.
   Motif scores label previous-token, induction, and first-token-sink patterns.
2. **Contribution (ATTR).** A head's pre-W_O slice, mapped through its W_O
   columns, is scored against the answer direction at the final position. The
   score uses the same frozen-norm convention as Lab 2, plus a second frozen
   norm when the architecture applies a post-attention norm before the head
   write reaches the residual stream.
3. **Causal role (CAUSAL).** Scoped head ablations ask whether removing one
   head changes the next-token logit gap. ``final_pos`` is the direct path;
   ``all_pos`` also removes earlier-position writes that later heads may read.

The course's recurring warning lives here in miniature: a beautiful heatmap is
not an explanation until it survives contribution scoring (using the same
frozen-norm linearization as Lab 2, plus per-block post-attention norm handling)
and a scoped intervention. Routing (attention pattern), writing (head output
projected through its W_O slice), and causal role (final_pos vs all_pos ablation)
are three different measurements that license three different strengths of claim.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping

import interp_bench as bench

LAB_ID = "L03"

CATEGORIES = ("synthetic", "cycle", "natural", "control")
REPEAT_CATEGORIES = ("synthetic", "cycle", "natural")

# Motif thresholds. Deliberately simple and visible; students are told the
# rule and invited to break it (see "label rule is not sacred" in the MD).
# Sink gets a HIGHER bar and LOWER priority: parking attention on position 0
# is the model's resting pattern (attention sink), so most heads carry large
# sink mass as background. The synth_digits example was added to demonstrate
# that the induction motif is structural (repeat copying) rather than lexical
# (letters or arithmetic sequences).
MOTIF_SCORE_BAR = 0.35
SINK_SCORE_BAR = 0.5
DIFFUSE_ENTROPY_FRAC = 0.85  # mean entropy / max possible entropy
NATURAL_CONFIRM_RATIO = 0.5

# Causal measurements are expensive on Tier B. The cap keeps the lab teachable;
# prompts are selected by baseline behavior first, not by file order.
BASELINE_SUCCESS_MARGIN = 0.0
ABLATION_EXAMPLE_CAP = 4
N_PREVIOUS_TOKEN_CANDIDATES = 2
N_SINK_CANDIDATES = 1
N_TOP_ATTRIBUTION_CANDIDATES = 2
N_RANDOM_CONTROLS = 2
N_LOW_ATTRIBUTION_CONTROLS = 1

# Direct logit attribution below this is mostly bf16 jitter. Rhos are reported
# both pooled and above this floor, so students see the measurement-noise goblin.
ATTRIBUTION_NOISE_FLOOR = 0.1


@dataclasses.dataclass(frozen=True)
class PatternPrompt:
    example_id: str
    category: str
    prompt: str
    target: str
    distractor: str
    note: str = ""


# Synthetic vocabularies avoid alphabet/number-line priors: "A B C A B" is a
# broken microscope because the alphabetic continuation equals the induction
# answer. "B F Q" has no such prior. Each synthetic prompt repeats its cycle
# so induction has multiple query positions to act on.
ALL_EXAMPLES: tuple[PatternPrompt, ...] = (
    PatternPrompt(
        "synth_letters",
        "synthetic",
        "B F Q B F Q B F",
        " Q",
        " B",
        "Non-alphabetical letters: no sequence prior to confound induction.",
    ),
    PatternPrompt("synth_colors", "synthetic", "red blue green red blue green red blue", " green", " red"),
    PatternPrompt("synth_animals", "synthetic", "dog cat bird dog cat bird dog cat", " bird", " dog"),
    # More varied to pop "induction is a general copying motif, not just letters"
    PatternPrompt("synth_digits", "synthetic", "1 7 3 1 7 3 1 7", " 3", " 1", "Numeric repeat to show the motif is structural, not lexical."),
    PatternPrompt(
        "synth_numbers",
        "synthetic",
        "seven three nine seven three nine seven three",
        " nine",
        " seven",
        "Number words in a non-arithmetic order.",
    ),
    PatternPrompt("synth_fruit", "synthetic", "apple pear banana apple pear banana apple pear", " banana", " apple", "Another vocab to show structural copying."),
    PatternPrompt("synth_shapes", "synthetic", "circle square star circle square star circle square", " star", " circle"),
    PatternPrompt(
        "cycle_moon",
        "cycle",
        "moon star moon star moon star moon",
        " star",
        " moon",
        "Period-2 cycle: previous-token and induction predictions coincide less cleanly.",
    ),
    PatternPrompt("cycle_sun", "cycle", "sun rain sun rain sun rain sun", " rain", " sun"),
    PatternPrompt("cycle_day", "cycle", "sun moon sun moon sun moon sun", " moon", " sun"),
    PatternPrompt(
        "nat_lab",
        "natural",
        "Marcus went to the lab. Olivia went to the",
        " lab",
        " store",
        "Natural-text induction: repeated phrase, new subject.",
    ),
    PatternPrompt("nat_door", "natural", "The wizard opened the ancient door. Behind the ancient", " door", " wall"),
    PatternPrompt(
        "nat_apples",
        "natural",
        "She bought fresh apples at the market. Everyone loved the fresh",
        " apples",
        " fruit",
    ),
    PatternPrompt("nat_train", "natural", "The train arrived at the station. Passengers left the", " station", " train"),
    PatternPrompt(
        "ctrl_fox",
        "control",
        "The quick brown fox jumps over the lazy",
        " dog",
        " cat",
        "No intentional repeats: induction-pattern scores should be blank or tiny here.",
    ),
    PatternPrompt(
        "ctrl_paris",
        "control",
        "The Eiffel Tower is in",
        " Paris",
        " Rome",
        "Factual recall, no repetition: a head strongly 'inducting' here is a label failure.",
    ),
    PatternPrompt(
        "ctrl_river",
        "control",
        "The river flows into the sea near the old",
        " city",
        " bridge",
        "Natural but no repeat structure for the motif.",
    ),
)

SMALL_SET_IDS = ("synth_colors", "cycle_moon", "nat_lab", "ctrl_fox")
MEDIUM_SET_IDS = SMALL_SET_IDS + ("synth_letters", "synth_animals", "nat_door", "ctrl_paris", "synth_fruit", "cycle_day")


# ---------------------------------------------------------------------------
# Prompt loading and validation
# ---------------------------------------------------------------------------


def validate_prompt_schema(examples: list[PatternPrompt]) -> None:
    seen: set[str] = set()
    for ex in examples:
        if not ex.example_id or not ex.prompt:
            raise ValueError(f"Example {ex!r} is missing an id or prompt.")
        if ex.example_id in seen:
            raise ValueError(f"Duplicate example_id {ex.example_id!r}.")
        seen.add(ex.example_id)
        if ex.category not in CATEGORIES:
            raise ValueError(f"Example {ex.example_id!r}: unknown category {ex.category!r}.")
        if not ex.target or not ex.distractor:
            raise ValueError(f"Example {ex.example_id!r} needs a target and a distractor.")


def interleave_by_category(examples: list[PatternPrompt]) -> list[PatternPrompt]:
    queues: dict[str, list[PatternPrompt]] = {cat: [] for cat in CATEGORIES}
    for ex in examples:
        queues.setdefault(ex.category, []).append(ex)
    out: list[PatternPrompt] = []
    while any(queues.values()):
        for cat in queues:
            if queues[cat]:
                out.append(queues[cat].pop(0))
    return out


def load_custom_prompt_set(path: pathlib.Path) -> list[PatternPrompt]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(
            f"Could not read prompt set {str(path)!r}: {exc}. "
            "--prompt-set must be one of small | medium | full, or a path to a prompts .json file."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse prompt JSON at {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError("Custom prompt file must be a JSON list of objects.")
    allowed = {f.name for f in dataclasses.fields(PatternPrompt)}
    out: list[PatternPrompt] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt item {i} is not an object: {item!r}")
        extra = set(item) - allowed
        if extra:
            raise ValueError(f"Prompt item {i} has unknown keys: {sorted(extra)}")
        out.append(PatternPrompt(**item))
    return out


def build_prompt_set(args: Any) -> list[PatternPrompt]:
    if args.prompt_set == "full":
        examples = list(ALL_EXAMPLES)
    elif args.prompt_set == "medium":
        examples = [ex for ex in ALL_EXAMPLES if ex.example_id in MEDIUM_SET_IDS]
    elif args.prompt_set == "small":
        examples = [ex for ex in ALL_EXAMPLES if ex.example_id in SMALL_SET_IDS]
    else:
        examples = load_custom_prompt_set(pathlib.Path(args.prompt_set))
    validate_prompt_schema(examples)
    examples = interleave_by_category(examples)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    return examples


# ---------------------------------------------------------------------------
# Motif scores
# ---------------------------------------------------------------------------
#
# Scores are means of attention mass over query positions where the motif is
# defined. A score of 0.8 reads as "80% of this head's attention goes where the
# motif says, where the motif says anything at all".


def prev_token_score(pattern: Any) -> float:
    """Mean attention to position q-1, over q >= 1. pattern: [seq, seq]."""
    seq = pattern.shape[0]
    if seq < 2:
        return 0.0
    return float(statistics.fmean(float(pattern[q, q - 1]) for q in range(1, seq)))


def first_token_score(pattern: Any) -> float:
    """Mean attention to position 0 over q >= 1.

    This is the attention-sink/resting pattern. It lands on BOS when a tokenizer
    prepends BOS and on the first visible token otherwise.
    """
    seq = pattern.shape[0]
    if seq < 2:
        return 0.0
    return float(statistics.fmean(float(pattern[q, 0]) for q in range(1, seq)))


def induction_targets(input_ids: list[int]) -> dict[int, list[int]]:
    """Induction targets for each query position.

    For query q, look for previous occurrences j < q of the SAME token. The
    classic induction key is the token after that previous occurrence, j+1.
    We require j+1 < q so adjacent repeats do not count the query token itself
    as its own "after previous occurrence" evidence.
    """
    targets: dict[int, list[int]] = {}
    for q, tok in enumerate(input_ids):
        hits = [j + 1 for j in range(q) if input_ids[j] == tok and j + 1 < q]
        if hits:
            targets[q] = hits
    return targets


def induction_score(pattern: Any, input_ids: list[int]) -> float | None:
    """Mean attention mass on induction targets.

    Returns None when the motif is never defined. Control prompts should land
    here; a numeric induction score on a control prompt is a prompt-audit event.
    """
    targets = induction_targets(input_ids)
    if not targets:
        return None
    masses = []
    for q, hits in targets.items():
        masses.append(float(sum(float(pattern[q, j]) for j in hits)))
    return float(statistics.fmean(masses))


def induction_coverage(input_ids: list[int]) -> dict[str, Any]:
    targets = induction_targets(input_ids)
    repeated_positions = sum(1 for i, tok in enumerate(input_ids) if tok in set(input_ids[:i]))
    n = len(input_ids)
    return {
        "prompt_n_tokens": n,
        "n_unique_token_ids": len(set(input_ids)),
        "n_repeated_positions": repeated_positions,
        "n_induction_query_positions": len(targets),
        "induction_query_frac": round(len(targets) / max(n, 1), 4),
        "has_induction_targets": bool(targets),
    }


def attention_entropy_bits(pattern: Any) -> tuple[float, float]:
    """Return (mean entropy in bits, mean entropy as fraction of causal max)."""
    seq = pattern.shape[0]
    ents, fracs = [], []
    for q in range(1, seq):
        row = pattern[q, : q + 1]
        ent = 0.0
        for v in row.tolist():
            if v > 0:
                ent -= v * math.log2(v)
        ents.append(ent)
        fracs.append(ent / math.log2(q + 1))
    if not ents:
        return 0.0, 0.0
    return float(statistics.fmean(ents)), float(statistics.fmean(fracs))


def label_head(prev: float, induct: float | None, first: float, entropy_frac: float) -> str:
    """Transparent motif label, intentionally simple and debatable."""
    content = {"previous_token": prev, "induction": induct if induct is not None else 0.0}
    best = max(content, key=lambda k: content[k])
    if content[best] >= MOTIF_SCORE_BAR:
        return best
    if first >= SINK_SCORE_BAR:
        return "first_token_sink"
    if entropy_frac >= DIFFUSE_ENTROPY_FRAC:
        return "diffuse"
    return "other"


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _fmean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return float(statistics.fmean(vals))


def _round_or_blank(value: float | None, digits: int = 4) -> float | str:
    return "" if value is None else round(float(value), digits)


def _rank(values: list[float]) -> list[float]:
    """Average ranks for ties; rank 1 is smallest value."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def spearman_rho(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _rank(xs), _rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in rx))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ry))
    if den_x == 0 or den_y == 0:
        return None
    return float(num / (den_x * den_y))


def token_rank(logits: Any, token_id: int) -> int:
    return int((logits > logits[token_id]).sum()) + 1


# ---------------------------------------------------------------------------
# Head attribution (frozen-norm, composing Lab 2's convention)
# ---------------------------------------------------------------------------


def _norm_kind(norm: Any) -> str:
    name = type(norm).__name__.lower()
    # Olmo/Llama/Gemma/Qwen RMSNorm variants all advertise rms in the class
    # name. Plain LayerNorm centers; RMSNorm does not.
    return "rms" if "rms" in name else "centered"


def frozen_norm_direction(norm: Any, x_ref: Any, downstream_direction: Any) -> tuple[Any, float, str]:
    """Linear direction through a frozen normalization layer.

    For RMSNorm, the frozen map is ``gain * x / rms(x_ref)``. For centered
    LayerNorm, the frozen map is ``gain * (x - mean(x)) / std(x_ref)``. Biases
    are constants and therefore not assigned to any head.
    """
    import torch

    kind = _norm_kind(norm)
    gain = getattr(norm, "weight", None)
    if gain is None:
        gain_vec = torch.ones_like(x_ref, dtype=torch.float32)
    else:
        gain_vec = gain.detach().to("cpu", torch.float32)
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5)))
    if kind == "rms":
        scale = 1.0 / float(torch.sqrt(x_ref.pow(2).mean() + eps))
        return scale * gain_vec * downstream_direction, scale, kind
    var = float(x_ref.var(unbiased=False))
    scale = 1.0 / float((var + eps) ** 0.5)
    v = scale * gain_vec * downstream_direction
    return v - v.mean() * torch.ones_like(v), scale, kind


def head_attribution_scores(
    bundle: bench.ModelBundle,
    comp_anatomy: bench.ComponentAnatomy,
    head_anatomy: bench.HeadAnatomy,
    att: bench.AttentionCapture,
    target_id: int,
    distractor_id: int,
) -> dict[str, Any]:
    """Score every head's final-position write against the answer direction.

    Two frozen linearizations compose here (consistent with Lab 2's DLA convention,
    plus one extra for post-attention norms):

    1. The final norm is frozen at the actual final stream (pre-final-norm
       residual, same as Labs 1–2).
    2. On post-norm architectures (Olmo-style), each block's attention output
       passes through a per-block norm before joining the stream. That norm is
       frozen at the block's full attention output. This helper handles both
       RMSNorm and centered LayerNorm.

    The projection bias and norm bias are not assigned to any head. They are
    reported as residual accounting terms (see head_attribution_accounting.csv)
    rather than stuffed into a convenient head-shaped costume. The per-head
    decomposition check (head_decomposition_check.json) must pass before any
    attribution number is interpreted.
    """
    import torch

    x_final = att.capture.streams[-1, -1]
    w_u = bundle.lm_head.weight
    direction = (w_u[target_id].detach() - w_u[distractor_id].detach()).to("cpu", torch.float32)

    final_w, s_final, final_kind = frozen_norm_direction(bundle.final_norm, x_final, direction)

    n_layers = bundle.anatomy.n_layers
    n_heads = head_anatomy.n_heads
    scores = [[0.0] * n_heads for _ in range(n_layers)]
    block_norm_kinds: list[str] = []
    block_norm_scales: list[float | None] = []
    for layer in range(n_layers):
        if comp_anatomy.attn_source == "post_norm":
            post = bench.get_by_path(bundle.blocks[layer], comp_anatomy.attn_hook_path)
            block_w, block_scale, block_kind = frozen_norm_direction(post, att.attn_out_last[layer], final_w)
            block_norm_kinds.append(block_kind)
            block_norm_scales.append(block_scale)
        else:
            block_w = final_w
            block_norm_kinds.append("identity")
            block_norm_scales.append(None)
        for head in range(n_heads):
            contrib = bench.head_contribution(bundle, head_anatomy, layer, head, att.o_in_last[layer])
            scores[layer][head] = float(contrib @ block_w)
    return {
        "scores": scores,
        "direction_vector": final_w,
        "frozen_final_scale": s_final,
        "final_norm_kind": final_kind,
        "block_norm_kinds": block_norm_kinds,
        "block_norm_scales": block_norm_scales,
    }


def write_head_attribution_accounting(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    comp_anatomy: bench.ComponentAnatomy,
    attr: dict[str, Any],
    first_comp: bench.ComponentCapture,
) -> None:
    """Diagnostic: compare summed head attribution with block attribution.

    This does not abort. Bias terms and frozen post-norm linearization make the
    head-sum an accounting approximation, not an identity. The artifact tells
    students where the approximation is large enough to remember.
    """
    rows = []
    w = attr["direction_vector"]
    max_abs_resid = 0.0
    for layer in range(bundle.anatomy.n_layers):
        head_sum = float(sum(attr["scores"][layer]))
        block_score = float(first_comp.attn_contrib[layer] @ w)
        residual = block_score - head_sum
        max_abs_resid = max(max_abs_resid, abs(residual))
        rows.append(
            {
                "layer": layer,
                "attn_source": comp_anatomy.attn_source,
                "block_score_exact_under_final_frozen_norm": round(block_score, 6),
                "sum_head_scores": round(head_sum, 6),
                "unassigned_bias_or_norm_residual": round(residual, 6),
                "abs_residual": round(abs(residual), 6),
            }
        )
    path = ctx.path("diagnostics", "head_attribution_accounting.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(
        path,
        "diagnostic",
        "Per-layer comparison of exact attention-block attribution and summed head attributions.",
    )
    jpath = ctx.path("diagnostics", "head_attribution_accounting.json")
    bench.write_json(
        jpath,
        {
            "max_abs_unassigned_residual": max_abs_resid,
            "attn_source": comp_anatomy.attn_source,
            "explanation": (
                "Head scores exclude shared projection/norm bias. On post-norm models they also use a frozen "
                "local linearization of the block norm. Large residuals should be mentioned when interpreting "
                "per-head attribution as an accounting ledger."
            ),
        },
    )
    ctx.register_artifact(jpath, "diagnostic", "Summary of head-attribution residual accounting.")


# ---------------------------------------------------------------------------
# Aggregation tables
# ---------------------------------------------------------------------------


def category_head_summaries(example_head_rows: list[dict[str, Any]], n_layers: int, n_heads: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str], dict[str, list[float]]] = {}
    for row in example_head_rows:
        key = (int(row["layer"]), int(row["head"]), str(row["category"]))
        grouped.setdefault(
            key,
            {"prev": [], "induct": [], "first": [], "ent": [], "ent_frac": [], "attr": []},
        )
        grouped[key]["prev"].append(float(row["prev_token_score"]))
        grouped[key]["first"].append(float(row["first_token_score"]))
        grouped[key]["ent"].append(float(row["entropy_bits"]))
        grouped[key]["ent_frac"].append(float(row["entropy_frac"]))
        grouped[key]["attr"].append(float(row["target_attribution"]))
        if row["induction_score"] != "":
            grouped[key]["induct"].append(float(row["induction_score"]))

    out = []
    for layer in range(n_layers):
        for head in range(n_heads):
            for category in CATEGORIES:
                g = grouped.get((layer, head, category))
                if not g:
                    continue
                prev = _fmean(g["prev"])
                first = _fmean(g["first"])
                induct = _fmean(g["induct"])
                ent_frac = _fmean(g["ent_frac"])
                out.append(
                    {
                        "layer": layer,
                        "head": head,
                        "category": category,
                        "n_examples": len(g["prev"]),
                        "prev_token_score": _round_or_blank(prev),
                        "induction_score": _round_or_blank(induct),
                        "first_token_score": _round_or_blank(first),
                        "mean_entropy_bits": _round_or_blank(_fmean(g["ent"])),
                        "mean_entropy_frac": _round_or_blank(ent_frac),
                        "mean_target_attribution": _round_or_blank(_fmean(g["attr"])),
                        "pattern_label": label_head(prev or 0.0, induct, first or 0.0, ent_frac or 0.0),
                    }
                )
    return out


def control_induction_audit(example_head_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in example_head_rows:
        if r["category"] != "control" or r["induction_score"] == "":
            continue
        score = float(r["induction_score"])
        if score >= MOTIF_SCORE_BAR:
            rows.append(
                {
                    "example_id": r["example_id"],
                    "layer": r["layer"],
                    "head": r["head"],
                    "induction_score": r["induction_score"],
                    "pattern_label": r["pattern_label"],
                    "audit_note": "Control prompt produced an above-bar induction score; inspect tokens and repeats.",
                }
            )
    if not rows:
        rows.append(
            {
                "example_id": "ALL_CONTROLS",
                "layer": "",
                "head": "",
                "induction_score": "",
                "pattern_label": "",
                "audit_note": f"No control-prompt head had induction_score >= {MOTIF_SCORE_BAR}.",
            }
        )
    return rows


def choose_ablation_candidates(
    head_table: list[dict[str, Any]], args: Any
) -> list[dict[str, Any]]:
    """Select motif, attribution, random, and low-attribution controls."""

    def numeric(row: Mapping[str, Any], key: str) -> float:
        v = row.get(key, 0.0)
        return 0.0 if v == "" or v is None else float(v)

    chosen: dict[tuple[int, int], dict[str, Any]] = {}

    def add(rows: list[dict[str, Any]], source: str, limit: int) -> None:
        for rank, row in enumerate(rows[: max(0, limit)], start=1):
            key = (int(row["layer"]), int(row["head"]))
            if key not in chosen:
                chosen[key] = dict(row)
                chosen[key]["candidate_sources"] = []
            chosen[key]["candidate_sources"].append(f"{source}:rank{rank}")

    by_label = lambda lab: [r for r in head_table if r["pattern_label"] == lab]  # noqa: E731
    add(sorted(by_label("induction"), key=lambda r: -numeric(r, "induction_score")), "motif_induction", args.ablate_top)
    add(sorted(by_label("previous_token"), key=lambda r: -numeric(r, "prev_token_score")), "motif_previous_token", N_PREVIOUS_TOKEN_CANDIDATES)
    add(sorted(by_label("first_token_sink"), key=lambda r: -numeric(r, "first_token_score")), "motif_sink", N_SINK_CANDIDATES)
    add(sorted(head_table, key=lambda r: -abs(numeric(r, "mean_target_attribution"))), "top_abs_attribution", N_TOP_ATTRIBUTION_CANDIDATES)

    rng = random.Random(args.seed)
    used = set(chosen)
    pool = [r for r in head_table if (int(r["layer"]), int(r["head"])) not in used]
    random_controls = rng.sample(pool, k=min(N_RANDOM_CONTROLS, len(pool)))
    add(random_controls, "random_control", len(random_controls))
    used = set(chosen)
    add(
        sorted(
            [r for r in head_table if (int(r["layer"]), int(r["head"])) not in used],
            key=lambda r: abs(numeric(r, "mean_target_attribution")),
        ),
        "low_abs_attribution_control",
        N_LOW_ATTRIBUTION_CONTROLS,
    )

    out = []
    for idx, row in enumerate(chosen.values(), start=1):
        row = dict(row)
        row["candidate_id"] = idx
        row["candidate_sources"] = ";".join(row["candidate_sources"])
        out.append(row)
    return out


def choose_ablation_examples(
    kept: list[tuple[PatternPrompt, int, int]], baseline_rows: list[dict[str, Any]]
) -> tuple[list[tuple[PatternPrompt, int, int]], str]:
    by_id = {r["example_id"]: r for r in baseline_rows}
    non_control = [(ex, t, d) for ex, t, d in kept if ex.category in REPEAT_CATEGORIES]
    successes = [
        item for item in non_control
        if float(by_id[item[0].example_id]["logit_diff_target_minus_distractor"]) >= BASELINE_SUCCESS_MARGIN
    ]
    if successes:
        ranked = sorted(
            successes,
            key=lambda item: float(by_id[item[0].example_id]["logit_diff_target_minus_distractor"]),
            reverse=True,
        )
        return ranked[:ABLATION_EXAMPLE_CAP], "baseline_success"
    ranked = sorted(
        non_control,
        key=lambda item: float(by_id[item[0].example_id]["logit_diff_target_minus_distractor"]),
        reverse=True,
    )
    return ranked[:ABLATION_EXAMPLE_CAP], "fallback_no_prompt_met_success_margin"


def summarize_ablation_rows(ablation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for r in ablation_rows:
        key = (int(r["layer"]), int(r["head"]))
        g = grouped.setdefault(
            key,
            {
                "layer": int(r["layer"]),
                "head": int(r["head"]),
                "pattern_label": r["pattern_label"],
                "candidate_sources": r.get("candidate_sources", ""),
                "attribution": [],
                "final_pos": [],
                "all_pos": [],
                "examples": set(),
            },
        )
        g["examples"].add(r["example_id"])
        if r["scope"] == "final_pos":
            g["final_pos"].append(float(r["causal_effect"]))
            g["attribution"].append(float(r["attribution_score"]))
        elif r["scope"] == "all_pos":
            g["all_pos"].append(float(r["causal_effect"]))
    out = []
    for g in grouped.values():
        direct = _fmean(g["final_pos"])
        total = _fmean(g["all_pos"])
        gap = None if direct is None or total is None else total - direct
        out.append(
            {
                "layer": g["layer"],
                "head": g["head"],
                "pattern_label": g["pattern_label"],
                "candidate_sources": g["candidate_sources"],
                "n_examples": len(g["examples"]),
                "mean_attribution_score": _round_or_blank(_fmean(g["attribution"])),
                "mean_direct_effect_final_pos": _round_or_blank(direct),
                "mean_total_effect_all_pos": _round_or_blank(total),
                "mean_indirect_gap_all_minus_final": _round_or_blank(gap),
                "abs_total_effect": _round_or_blank(abs(total) if total is not None else None),
            }
        )
    return sorted(out, key=lambda r: abs(float(r["mean_total_effect_all_pos"] or 0.0)), reverse=True)


def attribution_ablation_stats(
    ablation_rows: list[dict[str, Any]], ablation_summary: list[dict[str, Any]]
) -> dict[str, Any]:
    final_rows = [r for r in ablation_rows if r["scope"] == "final_pos"]
    xs = [float(r["attribution_score"]) for r in final_rows]
    ys = [float(r["causal_effect"]) for r in final_rows]
    signal = [(x, y) for x, y in zip(xs, ys) if abs(x) >= ATTRIBUTION_NOISE_FLOOR]

    head_rows = [r for r in ablation_summary if r["mean_direct_effect_final_pos"] != ""]
    hx = [float(r["mean_attribution_score"]) for r in head_rows]
    hy = [float(r["mean_direct_effect_final_pos"]) for r in head_rows]
    hsignal = [(x, y) for x, y in zip(hx, hy) if abs(x) >= ATTRIBUTION_NOISE_FLOOR]
    return {
        "per_example_rho": spearman_rho(xs, ys),
        "per_example_signal_rho": spearman_rho([p[0] for p in signal], [p[1] for p in signal]),
        "per_example_signal_n": len(signal),
        "by_head_rho": spearman_rho(hx, hy),
        "by_head_signal_rho": spearman_rho([p[0] for p in hsignal], [p[1] for p in hsignal]),
        "by_head_signal_n": len(hsignal),
        "n_final_pos_pairs": len(final_rows),
        "n_candidate_heads": len(head_rows),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_attention_heatmap_panel(
    ctx: bench.RunContext,
    att: bench.AttentionCapture,
    heads: list[tuple[int, int, str]],
    example_id: str,
) -> None:
    """Token-labeled attention heatmaps for showcase heads."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not heads:
        return
    labels = [bench.visible_token(t) for t in att.capture.tokens_text]
    n = len(heads)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.8), constrained_layout=True)
    if n == 1:
        axes = [axes]
    im = None
    for ax, (layer, head, title) in zip(axes, heads):
        pattern = np.array(att.attentions[layer, head])
        im = ax.imshow(pattern, vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(f"L{layer}H{head}\n{title}", fontsize=9)
        ax.set_xlabel("key position read from")
        ax.set_ylabel("query position reading")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, label="attention weight")
    fig.suptitle(f"Showcase heads on {example_id!r}: routing patterns, not explanations by themselves")
    bench.save_figure(
        ctx,
        fig,
        f"attention_heads_{bench.sanitize_tag(example_id)}.png",
        "Token-labeled attention patterns for motif and high-attribution heads.",
    )


def plot_motif_maps(ctx: bench.RunContext, head_rows: list[dict[str, Any]], n_layers: int, n_heads: int) -> None:
    """Four layer-by-head grids: motifs plus entropy."""
    import matplotlib.pyplot as plt
    import numpy as np

    grids = {
        "previous-token score": np.zeros((n_layers, n_heads)),
        "induction score": np.zeros((n_layers, n_heads)),
        "first-token sink score": np.zeros((n_layers, n_heads)),
        "entropy fraction": np.zeros((n_layers, n_heads)),
    }
    for r in head_rows:
        grids["previous-token score"][r["layer"], r["head"]] = r["prev_token_score"]
        grids["induction score"][r["layer"], r["head"]] = r["induction_score"] if r["induction_score"] != "" else 0.0
        grids["first-token sink score"][r["layer"], r["head"]] = r["first_token_score"]
        grids["entropy fraction"][r["layer"], r["head"]] = r["mean_entropy_frac"]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.5), sharey=True, constrained_layout=True)
    axes_flat = list(axes.ravel())
    for ax, (name, grid) in zip(axes_flat, grids.items()):
        im = ax.imshow(grid, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(name)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, fraction=0.035)
    fig.suptitle("Head motif map: routing patterns before contribution or causality")
    bench.save_figure(ctx, fig, "motif_maps.png", "Layer-by-head grids of motif scores and attention entropy.")


def plot_head_attribution_zoom(ctx: bench.RunContext, head_rows: list[dict[str, Any]], n_layers: int) -> None:
    """Lab 2 said which layers matter; this resolves those bars into heads."""
    fig, ax = bench.new_figure(figsize=(10.5, 5.8))
    by_layer: dict[int, list[float]] = {l: [] for l in range(n_layers)}
    for r in head_rows:
        by_layer[r["layer"]].append(float(r["mean_target_attribution"]))
    totals = [sum(v) for v in by_layer.values()]
    ax.bar(range(n_layers), totals, label="sum over heads")
    top = sorted(head_rows, key=lambda r: abs(float(r["mean_target_attribution"])), reverse=True)[:10]
    for r in top:
        marker = "*" if r["pattern_label"] == "induction" else "o"
        ax.scatter(r["layer"], r["mean_target_attribution"], s=70, marker=marker, zorder=3)
        ax.annotate(
            f"L{r['layer']}H{r['head']}\n{r['pattern_label']}",
            (r["layer"], r["mean_target_attribution"]),
            textcoords="offset points",
            xytext=(4, 5),
            fontsize=7,
            rotation=18,
        )
    ax.axhline(0, linewidth=0.6)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean direct attribution to target-minus-distractor")
    ax.set_title("Attention contribution resolved into heads (star = induction-labeled)")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "head_attribution_by_layer.png", "Per-layer attention attribution resolved into heads.")


def plot_direct_vs_indirect(ctx: bench.RunContext, ablation_rows: list[dict[str, Any]]) -> None:
    """The composition reveal: direct-path effect vs all-position effect."""
    pairs: dict[tuple, dict[str, Any]] = {}
    for r in ablation_rows:
        key = (r["example_id"], r["layer"], r["head"])
        pairs.setdefault(key, {"label": r["pattern_label"], "source": r.get("candidate_sources", "")})[r["scope"]] = float(r["causal_effect"])
    xs, ys, labels = [], [], []
    for d in pairs.values():
        if "final_pos" in d and "all_pos" in d:
            xs.append(d["final_pos"])
            ys.append(d["all_pos"])
            labels.append(d["label"])
    if not xs:
        return
    fig, ax = bench.new_figure(figsize=(7.8, 6.7))
    for lab in sorted(set(labels)):
        sel = [(x, y) for x, y, l in zip(xs, ys, labels) if l == lab]
        ax.scatter([p[0] for p in sel], [p[1] for p in sel], s=46, alpha=0.82, label=f"{lab} (n={len(sel)})")
    lim = max(max(abs(v) for v in xs), max(abs(v) for v in ys)) * 1.15 or 1.0
    ax.plot([-lim, lim], [-lim, lim], linewidth=0.8, linestyle="--", label="direct = all-position")
    ax.axhline(0, linewidth=0.5)
    ax.axvline(0, linewidth=0.5)
    ax.set_xlabel("direct effect: head zeroed at final position only")
    ax.set_ylabel("all-position effect: head zeroed everywhere")
    ax.set_title("Composition check: gap from the diagonal is indirect-path evidence")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "direct_vs_indirect_effect.png", "Direct-path vs all-position ablation effects.")


def plot_attribution_vs_ablation(
    ctx: bench.RunContext,
    ablation_rows: list[dict[str, Any]],
    ablation_summary: list[dict[str, Any]],
    stats: dict[str, Any],
) -> None:
    if not ablation_rows:
        return
    fig, ax = bench.new_figure(figsize=(7.8, 6.3))
    per = [r for r in ablation_rows if r["scope"] == "final_pos"]
    ax.scatter(
        [float(r["attribution_score"]) for r in per],
        [float(r["causal_effect"]) for r in per],
        s=26,
        alpha=0.28,
        label="per prompt",
    )
    agg = [r for r in ablation_summary if r["mean_direct_effect_final_pos"] != ""]
    ax.scatter(
        [float(r["mean_attribution_score"]) for r in agg],
        [float(r["mean_direct_effect_final_pos"]) for r in agg],
        s=72,
        marker="D",
        label="candidate mean",
    )
    for r in sorted(agg, key=lambda x: abs(float(x["mean_direct_effect_final_pos"])), reverse=True)[:6]:
        ax.annotate(
            f"L{r['layer']}H{r['head']}",
            (float(r["mean_attribution_score"]), float(r["mean_direct_effect_final_pos"])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    xs = [float(r["attribution_score"]) for r in per] + [float(r["mean_attribution_score"]) for r in agg]
    ys = [float(r["causal_effect"]) for r in per] + [float(r["mean_direct_effect_final_pos"]) for r in agg]
    lim = max(max(abs(v) for v in xs), max(abs(v) for v in ys)) * 1.15 or 1.0
    ax.plot([-lim, lim], [-lim, lim], linewidth=0.8, linestyle="--")
    ax.axvspan(-ATTRIBUTION_NOISE_FLOOR, ATTRIBUTION_NOISE_FLOOR, alpha=0.12, label="attribution noise floor")
    ax.axhline(0, linewidth=0.5)
    ax.axvline(0, linewidth=0.5)
    title = "Head attribution vs direct-path ablation"
    if stats.get("per_example_rho") is not None:
        title += f"\nper-prompt rho={stats['per_example_rho']:.3f}"
    if stats.get("by_head_rho") is not None:
        title += f"; by-head rho={stats['by_head_rho']:.3f}"
    ax.set_title(title)
    ax.set_xlabel("frozen-norm head attribution")
    ax.set_ylabel("direct-path causal effect")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "head_attribution_vs_ablation.png", "Attribution versus direct-path ablation.")


def plot_ablation_effect_by_head(ctx: bench.RunContext, ablation_summary: list[dict[str, Any]]) -> None:
    if not ablation_summary:
        return
    rows = sorted(ablation_summary, key=lambda r: abs(float(r["mean_total_effect_all_pos"] or 0.0)), reverse=True)[:16]
    fig, ax = bench.new_figure(figsize=(11.0, 5.8))
    labels = [f"L{r['layer']}H{r['head']}\n{r['pattern_label']}" for r in rows]
    xs = list(range(len(rows)))
    direct = [float(r["mean_direct_effect_final_pos"] or 0.0) for r in rows]
    total = [float(r["mean_total_effect_all_pos"] or 0.0) for r in rows]
    ax.scatter(xs, direct, marker="o", s=58, label="final_pos direct")
    ax.scatter(xs, total, marker="s", s=58, label="all_pos total")
    for x, d, t in zip(xs, direct, total):
        ax.plot([x, x], [d, t], linewidth=1.0, alpha=0.5)
    ax.axhline(0, linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean causal effect on logit gap")
    ax.set_title("Candidate heads: direct effect and indirect gap")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "ablation_effect_by_head.png", "Direct and all-position ablation effects by head.")


def plot_routing_to_causality(ctx: bench.RunContext, head_table: list[dict[str, Any]], ablation_summary: list[dict[str, Any]]) -> None:
    if not ablation_summary:
        return
    index = {(int(r["layer"]), int(r["head"])): r for r in head_table}
    rows = []
    for r in ablation_summary:
        ht = index[(int(r["layer"]), int(r["head"]))]
        rows.append((ht, r))
    fig, ax = bench.new_figure(figsize=(7.6, 6.2))
    for label in sorted(set(ht["pattern_label"] for ht, _ in rows)):
        pts = [(ht, ar) for ht, ar in rows if ht["pattern_label"] == label]
        ax.scatter(
            [float(ht["induction_score"] or 0.0) for ht, _ in pts],
            [float(ar["mean_total_effect_all_pos"] or 0.0) for _, ar in pts],
            s=[45 + min(120, 35 * abs(float(ht["mean_target_attribution"]))) for ht, _ in pts],
            alpha=0.75,
            label=label,
        )
    for ht, ar in sorted(rows, key=lambda p: abs(float(p[1]["mean_total_effect_all_pos"] or 0.0)), reverse=True)[:6]:
        ax.annotate(
            f"L{ht['layer']}H{ht['head']}",
            (float(ht["induction_score"] or 0.0), float(ar["mean_total_effect_all_pos"] or 0.0)),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    ax.axhline(0, linewidth=0.5)
    ax.axvline(MOTIF_SCORE_BAR, linewidth=0.8, linestyle="--", label="induction label bar")
    ax.set_xlabel("mean induction motif score")
    ax.set_ylabel("mean all-position ablation effect")
    ax.set_title("Routing score versus causal effect: motif label is not yet a circuit claim")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "routing_to_causality.png", "Induction motif score versus all-position causal effect.")


# ---------------------------------------------------------------------------
# Claims, card, and summary
# ---------------------------------------------------------------------------


def draft_claims(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    head_table: list[dict[str, Any]],
    ablation_summary: list[dict[str, Any]],
    natural_confirmations: list[dict[str, Any]],
    control_audit: list[dict[str, Any]],
) -> list[dict[str, str]]:
    run_name = ctx.run_dir.name
    claims: list[dict[str, str]] = []

    inducts = [r for r in head_table if r["pattern_label"] == "induction"]
    if inducts:
        top = max(inducts, key=lambda r: float(r["induction_score"] or 0.0))
        claims.append(
            {
                "id": f"{LAB_ID}-C1",
                "tag": "OBS",
                "text": (
                    f"{len(inducts)} of {len(head_table)} heads in {bundle.anatomy.model_id} were labeled induction "
                    f"by the stated motif rule; the strongest, L{top['layer']}H{top['head']}, put "
                    f"{float(top['induction_score']):.2f} of its attention mass on induction targets where the motif "
                    "was defined."
                ),
                "artifact": f"runs/{run_name}/tables/head_table.csv",
                "falsifier": "Fresh repeated-token families make the same heads fall below the induction bar.",
            }
        )
    if natural_confirmations:
        ok = [r for r in natural_confirmations if r["confirmed"]]
        verdict = "supports transfer beyond toy cycles" if 2 * len(ok) >= len(natural_confirmations) else "looks prompt-set-specific"
        claims.append(
            {
                "id": f"{LAB_ID}-C2",
                "tag": "OBS",
                "text": (
                    f"{len(ok)}/{len(natural_confirmations)} synthetic/cycle induction candidates retained at least "
                    f"{NATURAL_CONFIRM_RATIO:.0%} of their synthetic induction score on natural repeated phrases; this {verdict}."
                ),
                "artifact": f"runs/{run_name}/tables/natural_confirmation.csv",
                "falsifier": "Longer natural documents with distractor repeats break the motif correspondence.",
            }
        )
    prev_cases = [
        r for r in ablation_summary
        if r["pattern_label"] == "previous_token"
        and r["mean_direct_effect_final_pos"] != ""
        and r["mean_total_effect_all_pos"] != ""
        and abs(float(r["mean_direct_effect_final_pos"])) < 0.35 * abs(float(r["mean_total_effect_all_pos"]))
        and abs(float(r["mean_total_effect_all_pos"])) > 0.2
    ]
    if prev_cases:
        r = max(prev_cases, key=lambda row: abs(float(row["mean_total_effect_all_pos"])))
        claims.append(
            {
                "id": f"{LAB_ID}-C3",
                "tag": "CAUSAL",
                "text": (
                    f"Previous-token head L{r['layer']}H{r['head']} shows an indirect-path signature: mean final-position "
                    f"ablation changes the answer logit gap by {float(r['mean_direct_effect_final_pos']):+.2f}, while all-position "
                    f"ablation changes it by {float(r['mean_total_effect_all_pos']):+.2f}. The head matters mainly through "
                    "earlier-position writes that later layers can read."
                ),
                "artifact": f"runs/{run_name}/tables/head_ablation_summary.csv",
                "falsifier": "Path patching fails to route the effect through candidate induction heads; the all-position effect was diffuse.",
            }
        )
    control_violations = [r for r in control_audit if r.get("example_id") != "ALL_CONTROLS"]
    if control_violations:
        claims.append(
            {
                "id": f"{LAB_ID}-C4",
                "tag": "OBS-CAVEAT",
                "text": (
                    f"{len(control_violations)} control-prompt head/example pairs crossed the induction-score bar. "
                    "The motif rule should be audited before treating induction labels as stable head types."
                ),
                "artifact": f"runs/{run_name}/diagnostics/control_induction_audit.csv",
                "falsifier": "Token inspection shows the control prompts contain real repeated-token induction targets.",
            }
        )
    return claims


def render_attention_routing_card(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    head_table: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    ablation_summary: list[dict[str, Any]],
    ablation_stats: dict[str, Any],
    natural_confirmations: list[dict[str, Any]],
    control_audit: list[dict[str, Any]],
) -> str:
    label_counts: dict[str, int] = {}
    for r in head_table:
        label_counts[r["pattern_label"]] = label_counts.get(r["pattern_label"], 0) + 1
    successes = [r for r in baseline_rows if r["category"] in REPEAT_CATEGORIES and r["target_beats_distractor"]]
    top_total = ablation_summary[0] if ablation_summary else None
    ok_nat = sum(1 for r in natural_confirmations if r["confirmed"])
    control_violations = [r for r in control_audit if r.get("example_id") != "ALL_CONTROLS"]

    lines = [
        "# Lab 3 attention-routing card",
        "",
        "## Scope",
        "",
        f"- model: `{bundle.anatomy.model_id}`",
        f"- prompt set: `{ctx.args.prompt_set}`; {len(baseline_rows)} single-token-gated examples",
        "- intervention: zero one head's pre-output-projection slice, either at the final position or all positions",
        "- evidence ladder: OBS motif → ATTR direct write score → CAUSAL scoped ablation",
        "",
        "## Baseline behavior",
        "",
        f"- repeat-bearing prompts where target beats distractor: {len(successes)}/{sum(1 for r in baseline_rows if r['category'] in REPEAT_CATEGORIES)}",
        "- prompts that fail this gate are still useful for pattern observation, but weak evidence for causal behavior.",
        "",
        "## Head labels",
        "",
        "- " + ", ".join(f"{k}: {v}" for k, v in sorted(label_counts.items())),
        "",
        "## Transfer and controls",
        "",
        f"- natural confirmation: {ok_nat}/{len(natural_confirmations)} synthetic/cycle candidates retained the motif" if natural_confirmations else "- natural confirmation: no above-bar synthetic/cycle candidates to test",
        f"- control induction audit: {len(control_violations)} above-bar control cases",
        "",
        "## Causal candidates",
        "",
    ]
    if top_total:
        lines.append(
            f"- strongest total-effect candidate: L{top_total['layer']}H{top_total['head']} "
            f"({top_total['pattern_label']}), direct {top_total['mean_direct_effect_final_pos']}, "
            f"all-position {top_total['mean_total_effect_all_pos']}, indirect gap {top_total['mean_indirect_gap_all_minus_final']}"
        )
    else:
        lines.append("- no ablations were run (`--ablate-top 0` or no candidates).")
    lines += [
        "",
        "## Attribution versus causal effect",
        "",
        f"- per-prompt Spearman rho: {ablation_stats.get('per_example_rho')}",
        f"- by-head Spearman rho: {ablation_stats.get('by_head_rho')}",
        f"- above-noise by-head rho: {ablation_stats.get('by_head_signal_rho')} (n={ablation_stats.get('by_head_signal_n')})",
        "",
        "## Non-claims",
        "",
        "- A motif label is not a permanent job title for a head.",
        "- Zero ablation is a causal stress test, not an in-distribution replacement. Lab 6 switches to mean-ablation for circuit claims.",
        "- The all-position effect bundles many possible indirect paths; it suggests composition but does not localize an edge.",
    ]
    return "\n".join(lines) + "\n"


def render_summary(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    head_table: list[dict[str, Any]],
    label_counts: dict[str, int],
    baseline_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    ablation_summary: list[dict[str, Any]],
    ablation_stats: dict[str, Any],
    natural_confirmations: list[dict[str, Any]],
    dropped: int,
    n_examples: int,
    claims: list[dict[str, str]],
) -> str:
    a = bundle.anatomy
    args = ctx.args
    repeat_n = sum(1 for r in baseline_rows if r["category"] in REPEAT_CATEGORIES)
    repeat_success = sum(1 for r in baseline_rows if r["category"] in REPEAT_CATEGORIES and r["target_beats_distractor"])
    top_attr = sorted(head_table, key=lambda r: abs(float(r["mean_target_attribution"])), reverse=True)[:5]
    top_total = ablation_summary[0] if ablation_summary else None

    lines = [
        "# Lab 3 run summary: attention routing and head motifs",
        "",
        "## Run identity",
        "",
        f"- model: `{a.model_id}` ({a.n_layers} blocks x {head_table[-1]['head'] + 1} heads)",
        f"- dtype: `{args.dtype}` | attention implementation: `{args.attn_implementation}` (patterns require eager)",
        f"- examples: {n_examples} kept, {dropped} dropped at the single-token answer gate",
        "- evidence levels: OBS (motifs), ATTR (head attribution), CAUSAL (scoped ablations)",
        "- self-checks: hook parity, lens self-check, component anatomy, DLA decomposition, head decomposition",
        "",
        "## 1. What behavior was studied?",
        "",
        f"Next-token copying under repetition, plus non-repeating controls. The model preferred the target over the distractor on {repeat_success}/{repeat_n} repeat-bearing prompts.",
        "",
        "## 2. What internal object was measured?",
        "",
        "Every head's attention pattern (routing/OBS), its final-position write scored against the answer direction under the same frozen-norm linearization as Lab 2 (contribution/ATTR), and its causal effect under two ablation scopes (final_pos = direct path; all_pos = includes upstream writes that later layers read / CAUSAL). These are deliberately kept as three separate measurements.",
        "",
        "## 3. What intervention or control was used?",
        "",
        "The intervention zeroes one head's pre-output-projection slice. `final_pos` measures the direct path; `all_pos` also removes earlier writes that later heads can read. Random and low-attribution heads act as controls, and no-repeat prompts audit false induction labels.",
        "",
        "## 4. Headline numbers",
        "",
        "- head labels: " + ", ".join(f"{k}: {v}" for k, v in sorted(label_counts.items())),
        "- top heads by |attribution|: " + ", ".join(
            f"L{r['layer']}H{r['head']} ({r['pattern_label']}, {float(r['mean_target_attribution']):+.2f})" for r in top_attr
        ),
    ]
    if top_total:
        lines.append(
            f"- strongest all-position ablation candidate: L{top_total['layer']}H{top_total['head']} "
            f"({top_total['pattern_label']}), direct {top_total['mean_direct_effect_final_pos']}, "
            f"all-position {top_total['mean_total_effect_all_pos']}"
        )
    if ablation_stats.get("per_example_rho") is not None:
        lines.append(f"- Spearman rho, attribution vs direct-path ablation (per prompt): {ablation_stats['per_example_rho']:.3f}")
    if ablation_stats.get("by_head_rho") is not None:
        lines.append(f"- Spearman rho, attribution vs direct-path ablation (by candidate head): {ablation_stats['by_head_rho']:.3f}")
    if natural_confirmations:
        ok = sum(1 for r in natural_confirmations if r["confirmed"])
        lines.append(f"- induction heads confirmed on natural text: {ok}/{len(natural_confirmations)}")
    lines += [
        "",
        "## 5. What claim is supported, and at what evidence level?",
        "",
    ]
    for c in claims:
        lines.append(f"- `{c['id']}` {c['tag']}: {c['text']}")
        lines.append(f"  - falsifier: {c['falsifier']}")
    lines += [
        "",
        "## 6. The reading order",
        "",
        "1. `attention_routing_card.md` — the one-page verdict and non-claims.",
        "2. `diagnostics/prompt_motif_coverage.csv` and `tables/baseline_behavior.csv` — make sure the microscope had repeated tokens and behavior to study.",
        "3. `plots/motif_maps.png` — where named routing patterns live.",
        "4. `plots/attention_heads_*.png` — what selected heads read on real tokens.",
        "5. `plots/head_attribution_by_layer.png` — which heads write toward the answer direction.",
        "6. `tables/head_ablation_summary.csv` and `plots/ablation_effect_by_head.png` — causal effects by head and scope.",
        "7. `plots/direct_vs_indirect_effect.png` and `plots/routing_to_causality.png` — the composition detector (all_pos >> final_pos) and motif-vs-effect scatter. Points off the diagonal or high-motif/low-causal are the payload.",
        "8. `diagnostics/control_induction_audit.csv` — false-positive audit for the motif rule (synthetic labels are prompt-set-specific until proven on natural text).",
        "",
        "## 7. Caveats students must carry forward",
        "",
        "- A motif label is a description of a pattern on *this* prompt distribution (synthetic/cycle vs natural), not a job title. The label rule is deliberately simple and debatable; finding where it fails is part of the lab.",
        "- High attention (OBS) is not high contribution (ATTR); high contribution is not guaranteed to have causal effect (CAUSAL). Compare motif_maps.png directly with head_attribution and the ablation scatter.",
        "- `all_pos` ablation bundles indirect paths (earlier writes that later heads can read). The gap vs final_pos is a composition signal, not a localized edge. Lab 5 (patching) and Lab 6 (circuits) are needed for stronger claims.",
        "- Zeroing a head is a causal stress test (scoped, direct vs indirect), but it is not the dataset-mean ablation used for faithful circuit cards in later labs.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    args = ctx.args
    examples = build_prompt_set(args)
    print(f"[lab3] prompt set: {len(examples)} examples")

    tokenizer = bundle.tokenizer
    kept: list[tuple[PatternPrompt, int, int]] = []
    gate_rows = []
    motif_coverage_rows = []
    for ex in examples:
        t_ids = tokenizer.encode(ex.target, add_special_tokens=False)
        d_ids = tokenizer.encode(ex.distractor, add_special_tokens=False)
        p_ids = tokenizer.encode(ex.prompt, add_special_tokens=False)
        ok = len(t_ids) == 1 and len(d_ids) == 1
        coverage = induction_coverage(p_ids)
        token_text = [bench.visible_token(tokenizer.decode([i])) for i in p_ids]
        gate_rows.append(
            {
                "example_id": ex.example_id,
                "category": ex.category,
                "kept": ok,
                "target": bench.visible_token(ex.target),
                "target_n_tokens": len(t_ids),
                "distractor": bench.visible_token(ex.distractor),
                "distractor_n_tokens": len(d_ids),
                "prompt_n_tokens": coverage["prompt_n_tokens"],
                "n_induction_query_positions": coverage["n_induction_query_positions"],
                "note": ex.note,
            }
        )
        motif_coverage_rows.append(
            {
                "example_id": ex.example_id,
                "category": ex.category,
                "prompt": ex.prompt,
                "tokens": " | ".join(token_text),
                **coverage,
                "expected_repeat_bearing": ex.category in REPEAT_CATEGORIES,
                "audit_note": (
                    "No induction targets under tokenizer; motif scores will be blank."
                    if ex.category in REPEAT_CATEGORIES and not coverage["has_induction_targets"]
                    else ""
                ),
            }
        )
        if ok:
            kept.append((ex, t_ids[0], d_ids[0]))
        else:
            print(f"[lab3] dropping {ex.example_id}: target or distractor is multi-token")

    gate_path = ctx.path("diagnostics", "answer_tokenization.csv")
    bench.write_csv_with_context(ctx, gate_path, gate_rows)
    ctx.register_artifact(gate_path, "diagnostic", "Single-token gate for targets/distractors.")
    coverage_path = ctx.path("diagnostics", "prompt_motif_coverage.csv")
    bench.write_csv_with_context(ctx, coverage_path, motif_coverage_rows)
    ctx.register_artifact(coverage_path, "diagnostic", "Prompt tokenization and induction-target coverage audit.")

    if not kept:
        raise RuntimeError("Every example was dropped at the single-token answer gate.")
    dropped = len(examples) - len(kept)
    print(f"[lab3] running {len(kept)} examples ({dropped} dropped)")

    # Instrument verification before any science. The component decomposition
    # check is included because per-head attribution inherits Lab 2's component
    # bookkeeping; a bad DLA ledger would make all head scores decorative fog.
    first_prompt = kept[0][0].prompt
    bench.run_hook_parity_check(ctx, bundle, first_prompt)
    comp_anatomy = bench.resolve_component_anatomy(ctx, bundle, first_prompt, rel_tolerance=args.dla_tolerance)
    first_comp = bench.run_with_component_cache(bundle, first_prompt, comp_anatomy)
    bench.run_decomposition_check(ctx, bundle, first_comp, rel_tolerance=args.dla_tolerance)
    head_anatomy = bench.resolve_head_anatomy(ctx, bundle)
    first_att = bench.run_with_attention_cache(bundle, first_prompt)
    bench.run_lens_self_check(ctx, bundle, first_att.capture)
    bench.run_head_decomposition_check(ctx, bundle, head_anatomy, first_att, rel_tolerance=args.dla_tolerance)

    n_layers = bundle.anatomy.n_layers
    n_heads = head_anatomy.n_heads

    captures: dict[str, bench.AttentionCapture] = {}
    baseline_rows: list[dict[str, Any]] = []
    example_head_rows: list[dict[str, Any]] = []
    per_head_acc: dict[tuple[int, int], dict[str, list]] = {
        (l, h): {"prev": [], "induct": [], "first": [], "ent": [], "ent_frac": [], "attr": []}
        for l in range(n_layers) for h in range(n_heads)
    }
    natural_induct: dict[tuple[int, int], list[float]] = {}
    synth_induct: dict[tuple[int, int], list[float]] = {}
    first_attr_written = False

    for i, (ex, t_id, d_id) in enumerate(kept):
        att = first_att if ex.prompt == first_prompt else bench.run_with_attention_cache(bundle, ex.prompt)
        captures[ex.example_id] = att
        attr = head_attribution_scores(bundle, comp_anatomy, head_anatomy, att, t_id, d_id)
        if not first_attr_written:
            write_head_attribution_accounting(ctx, bundle, comp_anatomy, attr, first_comp)
            first_attr_written = True

        logits = att.capture.final_logits_last
        model_diff = float(logits[t_id] - logits[d_id])
        top_id = int(logits.argmax())
        coverage = induction_coverage(att.capture.input_ids)
        baseline_rows.append(
            {
                "example_id": ex.example_id,
                "category": ex.category,
                "prompt": ex.prompt,
                "target": bench.visible_token(ex.target),
                "distractor": bench.visible_token(ex.distractor),
                "logit_diff_target_minus_distractor": round(model_diff, 4),
                "target_rank": token_rank(logits, t_id),
                "distractor_rank": token_rank(logits, d_id),
                "target_beats_distractor": model_diff > 0,
                "top1_token": bench.visible_token(tokenizer.decode([top_id])),
                "top1_is_target": top_id == t_id,
                "usable_for_ablation_gate": ex.category in REPEAT_CATEGORIES and model_diff >= BASELINE_SUCCESS_MARGIN,
                **coverage,
            }
        )

        for l in range(n_layers):
            for h in range(n_heads):
                acc = per_head_acc[(l, h)]
                pattern = att.attentions[l, h]
                prev = prev_token_score(pattern)
                first = first_token_score(pattern)
                ind = induction_score(pattern, att.capture.input_ids)
                ent, ent_frac = attention_entropy_bits(pattern)

                acc["prev"].append(prev)
                acc["first"].append(first)
                if ind is not None:
                    acc["induct"].append(ind)
                    bucket = natural_induct if ex.category == "natural" else (
                        synth_induct if ex.category in ("synthetic", "cycle") else None
                    )
                    if bucket is not None:
                        bucket.setdefault((l, h), []).append(ind)
                acc["ent"].append(ent)
                acc["ent_frac"].append(ent_frac)
                acc["attr"].append(attr["scores"][l][h])
                example_head_rows.append(
                    {
                        "example_id": ex.example_id,
                        "category": ex.category,
                        "repeat_bearing_category": ex.category in REPEAT_CATEGORIES,
                        "n_induction_query_positions": coverage["n_induction_query_positions"],
                        "layer": l,
                        "head": h,
                        "prev_token_score": round(prev, 4),
                        "induction_score": round(ind, 4) if ind is not None else "",
                        "first_token_score": round(first, 4),
                        "entropy_bits": round(ent, 4),
                        "entropy_frac": round(ent_frac, 4),
                        "target_attribution": round(attr["scores"][l][h], 4),
                        "pattern_label": label_head(prev, ind, first, ent_frac),
                    }
                )

        traj = bench.compute_lens_trajectory(bundle, att.capture, target_id=t_id, distractor_id=d_id, topk=args.topk)
        bench.dump_example_state(ctx, bundle, ex.example_id, att.capture, traj, target=ex.target, distractor=ex.distractor)
        print(f"[lab3] [{i + 1}/{len(kept)}] {ex.example_id} logit_diff={model_diff:+.3f}")

    baseline_path = ctx.path("tables", "baseline_behavior.csv")
    bench.write_csv_with_context(ctx, baseline_path, baseline_rows)
    ctx.register_artifact(baseline_path, "table", "Per-prompt baseline behavior and motif coverage.")

    # The head table: one row per head, averaged over all kept prompts. Category
    # summaries are written separately so students can spot natural/control splits.
    head_table: list[dict[str, Any]] = []
    for l in range(n_layers):
        for h in range(n_heads):
            acc = per_head_acc[(l, h)]
            prev = statistics.fmean(acc["prev"])
            first = statistics.fmean(acc["first"])
            induct = statistics.fmean(acc["induct"]) if acc["induct"] else None
            ent_frac = statistics.fmean(acc["ent_frac"])
            head_table.append(
                {
                    "layer": l,
                    "head": h,
                    "prev_token_score": round(prev, 4),
                    "induction_score": round(induct, 4) if induct is not None else "",
                    "first_token_score": round(first, 4),
                    "mean_entropy_bits": round(statistics.fmean(acc["ent"]), 4),
                    "mean_entropy_frac": round(ent_frac, 4),
                    "mean_target_attribution": round(statistics.fmean(acc["attr"]), 4),
                    "pattern_label": label_head(prev, induct, first, ent_frac),
                }
            )

    head_path = ctx.path("tables", "head_table.csv")
    bench.write_csv_with_context(ctx, head_path, head_table)
    ctx.register_artifact(head_path, "table", "Every head: motif scores, entropy, label, attribution.")
    example_head_path = ctx.path("tables", "example_head_scores.csv")
    bench.write_csv_with_context(ctx, example_head_path, example_head_rows)
    ctx.register_artifact(example_head_path, "table", "Per-example head motif scores and attribution.")
    category_head_path = ctx.path("tables", "head_scores_by_category.csv")
    bench.write_csv_with_context(ctx, category_head_path, category_head_summaries(example_head_rows, n_layers, n_heads))
    ctx.register_artifact(category_head_path, "table", "Head motif and attribution scores broken out by prompt category.")
    control_audit = control_induction_audit(example_head_rows)
    control_path = ctx.path("diagnostics", "control_induction_audit.csv")
    bench.write_csv_with_context(ctx, control_path, control_audit)
    ctx.register_artifact(control_path, "diagnostic", "No-repeat control prompts that nevertheless triggered induction scores.")

    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, example_head_rows)
    ctx.register_artifact(results_path, "results", "Alias of example_head_scores.csv for the standard run contract.")

    label_counts: dict[str, int] = {}
    for r in head_table:
        label_counts[r["pattern_label"]] = label_counts.get(r["pattern_label"], 0) + 1
    print("[lab3] head labels: " + ", ".join(f"{k}={v}" for k, v in sorted(label_counts.items())))

    # Natural-text confirmation of synthetic/cycle-labeled induction heads.
    natural_confirmations = []
    for (l, h), synth_scores in sorted(synth_induct.items(), key=lambda kv: -statistics.fmean(kv[1])):
        s_mean = statistics.fmean(synth_scores)
        if s_mean < MOTIF_SCORE_BAR:
            continue
        nat = natural_induct.get((l, h), [])
        n_mean = statistics.fmean(nat) if nat else 0.0
        natural_confirmations.append(
            {
                "layer": l,
                "head": h,
                "synthetic_cycle_induction_score": round(s_mean, 4),
                "natural_induction_score": round(n_mean, 4),
                "confirmation_ratio": round(n_mean / max(abs(s_mean), 1e-9), 4),
                "confirmed": bool(nat) and n_mean >= NATURAL_CONFIRM_RATIO * s_mean,
            }
        )
    nat_path = ctx.path("tables", "natural_confirmation.csv")
    bench.write_csv_with_context(ctx, nat_path, natural_confirmations or [{"note": "No above-bar synthetic/cycle induction candidates."}])
    ctx.register_artifact(nat_path, "table", "Do synthetic/cycle induction heads still induct on natural text?")

    # Scoped ablations: motif candidates + attribution candidates + controls.
    ablation_rows: list[dict[str, Any]] = []
    ablation_summary: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    ablation_examples: list[tuple[PatternPrompt, int, int]] = []
    ablation_example_policy = "not_run"
    if args.ablate_top > 0:
        candidates = choose_ablation_candidates(head_table, args)
        cand_path = ctx.path("tables", "ablation_candidate_manifest.csv")
        bench.write_csv_with_context(ctx, cand_path, candidates)
        ctx.register_artifact(cand_path, "table", "Why each head was selected for ablation.")

        kept_index = {ex.example_id: i for i, (ex, _, _) in enumerate(kept)}
        ablation_examples, ablation_example_policy = choose_ablation_examples(kept, baseline_rows)
        total = len(candidates) * 2 * len(ablation_examples)
        print(
            f"[lab3] ablating {len(candidates)} heads x 2 scopes x "
            f"{len(ablation_examples)} prompts = {total} forwards ({ablation_example_policy})"
        )
        baseline_by_id = {r["example_id"]: r for r in baseline_rows}
        for r in candidates:
            for ex, t_id, d_id in ablation_examples:
                base = captures[ex.example_id].capture.final_logits_last
                base_diff = float(base[t_id] - base[d_id])
                example_attr = per_head_acc[(int(r["layer"]), int(r["head"]))]["attr"][kept_index[ex.example_id]]
                for scope in ("final_pos", "all_pos"):
                    logits = bench.run_with_head_ablation(
                        bundle, ex.prompt, head_anatomy, int(r["layer"]), int(r["head"]), scope=scope
                    )
                    abl_diff = float(logits[t_id] - logits[d_id])
                    ablation_rows.append(
                        {
                            "example_id": ex.example_id,
                            "category": ex.category,
                            "baseline_gate_passed": baseline_by_id[ex.example_id]["usable_for_ablation_gate"],
                            "candidate_id": r["candidate_id"],
                            "candidate_sources": r["candidate_sources"],
                            "layer": int(r["layer"]),
                            "head": int(r["head"]),
                            "pattern_label": r["pattern_label"],
                            "scope": scope,
                            "intervention_site": "pre_WO_head_slice_zeroed",
                            "attribution_score": round(example_attr, 4),
                            "mean_attribution_score": r["mean_target_attribution"],
                            "base_logit_diff": round(base_diff, 4),
                            "ablated_logit_diff": round(abl_diff, 4),
                            "causal_effect": round(base_diff - abl_diff, 4),
                        }
                    )
        abl_path = ctx.path("tables", "head_ablation_results.csv")
        bench.write_csv_with_context(ctx, abl_path, ablation_rows)
        ctx.register_artifact(abl_path, "table", "Scoped head ablations: direct-path vs all-position effects.")
        ablation_summary = summarize_ablation_rows(ablation_rows)
        abl_summary_path = ctx.path("tables", "head_ablation_summary.csv")
        bench.write_csv_with_context(ctx, abl_summary_path, ablation_summary)
        ctx.register_artifact(abl_summary_path, "table", "Ablation effects averaged by head and scope.")

    manifest_path = ctx.path("diagnostics", "ablation_manifest.json")
    bench.write_json(
        manifest_path,
        {
            "ablation_ran": bool(ablation_rows),
            "intervention": "zero one head's pre-output-projection slice before W_O",
            "scopes": {
                "final_pos": "zero only at the final query position; direct path comparable to attribution",
                "all_pos": "zero at every position; includes earlier writes later layers may read",
            },
            "baseline_success_margin": BASELINE_SUCCESS_MARGIN,
            "ablation_example_cap": ABLATION_EXAMPLE_CAP,
            "ablation_example_policy": ablation_example_policy,
            "ablation_examples": [ex.example_id for ex, _, _ in ablation_examples],
            "n_candidates": len(candidates),
            "candidate_selection": {
                "top_induction": args.ablate_top,
                "previous_token": N_PREVIOUS_TOKEN_CANDIDATES,
                "sink": N_SINK_CANDIDATES,
                "top_abs_attribution": N_TOP_ATTRIBUTION_CANDIDATES,
                "random_controls": N_RANDOM_CONTROLS,
                "low_abs_attribution_controls": N_LOW_ATTRIBUTION_CONTROLS,
            },
            "caveat": "This is zero-ablation, not dataset-mean ablation; Lab 6 uses mean-ablation for circuit cards.",
        },
    )
    ctx.register_artifact(manifest_path, "diagnostic", "Exact definition and scope of the head-ablation intervention.")

    ablation_stats = attribution_ablation_stats(ablation_rows, ablation_summary)

    # Plots.
    if not args.no_plots:
        plot_motif_maps(ctx, head_table, n_layers, n_heads)
        showcase_id = args.showcase or next(
            (ex.example_id for ex, _, _ in kept if ex.category == "synthetic"), kept[0][0].example_id
        )
        if showcase_id in captures:
            def best_for(label: str, score_key: str) -> dict[str, Any] | None:
                rows = [r for r in head_table if r["pattern_label"] == label]
                if not rows:
                    return None
                return max(rows, key=lambda r: float(r[score_key] or 0.0))

            heatmap_heads: list[tuple[int, int, str]] = []
            for label, score_key, title in (
                ("induction", "induction_score", "strongest induction motif"),
                ("previous_token", "prev_token_score", "strongest previous-token motif"),
                ("first_token_sink", "first_token_score", "strongest sink motif"),
            ):
                row = best_for(label, score_key)
                if row is not None:
                    heatmap_heads.append((int(row["layer"]), int(row["head"]), title))
            top_attr_head = max(head_table, key=lambda r: abs(float(r["mean_target_attribution"])))
            top_key = (int(top_attr_head["layer"]), int(top_attr_head["head"]))
            if top_key not in {(l, h) for l, h, _ in heatmap_heads}:
                heatmap_heads.append((top_key[0], top_key[1], "top |attribution| head"))
            plot_attention_heatmap_panel(ctx, captures[showcase_id], heatmap_heads, showcase_id)
        else:
            print(f"[lab3] WARNING: --showcase {args.showcase!r} did not match any kept example id.")
        plot_head_attribution_zoom(ctx, head_table, n_layers)
        plot_direct_vs_indirect(ctx, ablation_rows)
        plot_attribution_vs_ablation(ctx, ablation_rows, ablation_summary, ablation_stats)
        plot_ablation_effect_by_head(ctx, ablation_summary)
        plot_routing_to_causality(ctx, head_table, ablation_summary)

    metrics = {
        "n_examples": len(kept),
        "n_dropped": dropped,
        "label_counts": label_counts,
        "baseline_repeat_prompt_successes": sum(
            1 for r in baseline_rows if r["category"] in REPEAT_CATEGORIES and r["target_beats_distractor"]
        ),
        "baseline_repeat_prompt_total": sum(1 for r in baseline_rows if r["category"] in REPEAT_CATEGORIES),
        "attribution_ablation_stats": ablation_stats,
        "attribution_noise_floor": ATTRIBUTION_NOISE_FLOOR,
        "n_ablations": len(ablation_rows),
        "n_candidate_heads_ablated": len(ablation_summary),
        "natural_confirmations": natural_confirmations,
        "control_induction_violations": [r for r in control_audit if r.get("example_id") != "ALL_CONTROLS"],
    }
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 3 metrics.")

    claims = draft_claims(ctx, bundle, head_table, ablation_summary, natural_confirmations, control_audit)
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)

    card = render_attention_routing_card(
        ctx, bundle, head_table, baseline_rows, ablation_summary, ablation_stats, natural_confirmations, control_audit
    )
    card_path = ctx.path("attention_routing_card.md")
    bench.write_text(card_path, card)
    ctx.register_artifact(card_path, "report", "One-page evidence card for the attention-routing claim.")

    summary = render_summary(
        ctx,
        bundle,
        head_table,
        label_counts,
        baseline_rows,
        ablation_rows,
        ablation_summary,
        ablation_stats,
        natural_confirmations,
        dropped,
        len(kept),
        claims,
    )
    summary_path = ctx.path("run_summary.md")
    bench.write_text(summary_path, summary)
    ctx.register_artifact(summary_path, "summary", "The seven standard questions answered with this run's numbers.")
    print(f"[lab3] wrote run_summary.md, attention_routing_card.md, and {len(claims)} drafted ledger claims")
