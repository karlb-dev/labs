"""Lab 11: Mechanistic reliability audit (capstone).

Given behavioral evidence and internal evidence, where should we trust this
model less, and what may we responsibly say?

This capstone is not another microscope trick.  It is a structured audit
harness: pick a narrow domain, collect behavioral and mechanistic evidence,
then write a report that keeps evidence rungs separate.  The harness assembles
numbers and scaffolding.  The student's work is the judgment: failure-mode
labels, ledger verdicts, safety-case prose, and the final recommendation.

Three domains are implemented end to end.  All follow the course capstone
contract: per example behavior + confidence proxy, logit-lens stabilization,
DLA summary, a causal intervention on a subset, one additional internal method,
and a manual failure-mode label.

* ``--audit-domain factual_qa`` (default): factual recall under paraphrase,
  continuing Lab 5's capital-fact setup on a base model.  The audit combines
  logit-lens preference stabilization, frozen-norm DLA, residual patching at
  two sites plus an unrelated-clean control, and a truth-direction monitor with
  held-out facts and a shuffled-label control.

* ``--audit-domain cot_faithfulness`` (recommended flagship): a fresh-slice
  replication of Lab 10 on a reasoning model.  It reuses Lab 10's hint and CoT
  load-bearing interventions, then adds a mechanistic method: a mass-mean
  "hint present" probe over residual streams at answer-emission time, with a
  family/item split and a shuffled-label control.

* ``--audit-domain sentiment_negation``: sentiment classification under
  minimal negation edits, on the tier's base model.  Every plain statement in
  ``data/affect_valence.csv`` has a minimally negated counterpart in
  ``data/affect_negation.csv`` whose mood label flips while the surface
  valence words stay put.  The audit combines plain-vs-negated pair-argmax
  behavior, lens stabilization, frozen-norm DLA, plain-into-negated residual
  patching at the final position (with an unrelated-plain control), and a
  Lab 4-style mass-mean valence probe trained on plain statements only,
  reported on held-out plain statements, the negated family (the headline:
  surface valence words vs composed meaning), and a shuffled-label control.

Evidence level: integration.  Individual rows remain OBS / ATTR / DECODE /
CAUSAL / SELF-REPORT.  No later section is allowed to borrow a stronger rung
than the artifact earned.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import inspect
import math
import pathlib
import re
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import interp_bench as bench

LAB_ID = "L11"

# ---------------------------------------------------------------------------
# Domain configuration
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, str] = {
    "base": "The capital of {X} is",
    "para_city": "The capital city of {X} is",
    "para_in": "In {X}, the capital is",
}

FACT_POOL: tuple[tuple[str, str, str], ...] = (
    # (id, subject, capital).  Distractors are chosen cyclically from this full
    # pool, not the selected subset, so --max-examples 1 cannot accidentally
    # make target == distractor.
    # Expanded for robustness (more continents, more variety).
    ("france", "France", "Paris"),
    ("germany", "Germany", "Berlin"),
    ("italy", "Italy", "Rome"),
    ("spain", "Spain", "Madrid"),
    ("japan", "Japan", "Tokyo"),
    ("russia", "Russia", "Moscow"),
    ("china", "China", "Beijing"),
    ("egypt", "Egypt", "Cairo"),
    ("greece", "Greece", "Athens"),
    ("poland", "Poland", "Warsaw"),
    ("austria", "Austria", "Vienna"),
    ("england", "England", "London"),
    # Expansion entries are restricted to single-word subjects and single-token
    # capitals (GPT-2-verified) so the tokenization gate keeps them instead of
    # silently dropping most of the "expansion".
    ("canada", "Canada", "Ottawa"),
    ("australia", "Australia", "Canberra"),
    ("switzerland", "Switzerland", "Bern"),
    ("turkey", "Turkey", "Ankara"),
    ("norway", "Norway", "Oslo"),
    ("denmark", "Denmark", "Copenhagen"),
    ("sweden", "Sweden", "Stockholm"),
    ("portugal", "Portugal", "Lisbon"),
    ("thailand", "Thailand", "Bangkok"),
    ("indonesia", "Indonesia", "Jakarta"),
)

FACTUAL_BUDGET_BY_TIER = {"a": 6, "b": 18, "c": 18}
COT_AUDIT_ITEMS_BY_TIER = {"a": 4, "b": 20, "c": 28}
COT_EXP2_ITEMS_BY_TIER = {"a": 2, "b": 24, "c": 36}  # synced to lab10 for consistent load-bearing stats

N_CAUSAL_SUBSET = 6
TRUTH_MONITOR_LAYER_FRACS = (0.4, 0.55, 0.7, 0.85)
COT_PROBE_LAYER_FRACS = (0.4, 0.55, 0.7, 0.85)
COT_FRESH_OFFSET = 1
COT_FRESH_STRIDE = 4

# Sentiment-under-negation domain.  The datasets are frozen and vendored:
# data/affect_valence.csv (Lab 4 follow-up family) plus a paired, minimally
# negated counterpart file whose mood labels flip while the valence words stay.
SENTIMENT_DATA_FILES = {"plain": "affect_valence.csv", "negated": "affect_negation.csv"}
SENTIMENT_QUESTION_SUFFIX = (
    "\nQuestion: Is the overall mood of that sentence positive or negative?"
    "\nAnswer:"
)
# Answer tokens for the two-way readout.  Both are single tokens under the
# course's base-model tokenizers (gpt2: 3967/4633; allenai/Olmo-3-1025-7B:
# 6928/8389); the runtime gate below re-verifies on whatever tokenizer is
# actually loaded and aborts on a multi-token split.
SENTIMENT_ANSWER_TEXT = {1: " positive", 0: " negative"}
SENTIMENT_BUDGET_BY_TIER = {"a": 6, "b": 48, "c": 48}  # source statements; each brings its negated twin
SENTIMENT_PROBE_LAYER_FRACS = TRUTH_MONITOR_LAYER_FRACS

LEDGER_ENTRY_RE = re.compile(
    r"^\[(?P<id>L\d{2}-C\d+)\]\s+"
    r"(?P<tag>OBS|ATTR|DECODE|CAUSAL|SELF-REPORT)\s*\|\s*(?P<text>.+)$"
)


@dataclasses.dataclass(frozen=True)
class AuditExample:
    fact_id: str
    template_id: str
    prompt: str
    subject: str
    target: str
    distractor: str
    target_id: int
    distractor_id: int
    subject_id: int
    subject_pos: int
    n_prompt_tokens: int
    corrupt_fact_id: str


@dataclasses.dataclass(frozen=True)
class SentimentExample:
    statement_id: str
    family: str          # "plain" or "negated"
    pair_id: str         # source plain statement_id shared by both halves
    statement: str
    label: int           # 1 = positive overall mood after composing negation
    prompt: str
    target: str          # answer token text for the true label
    distractor: str
    target_id: int
    distractor_id: int
    n_prompt_tokens: int
    meta: str


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def arg_value(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def token_pieces(tokenizer: Any, ids: Sequence[int]) -> str:
    if not ids:
        return ""
    try:
        return "|".join(tokenizer.convert_ids_to_tokens(list(ids)))
    except Exception:
        return "|".join(str(i) for i in ids)


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> list[int]:
    if not needle or len(needle) > len(haystack):
        return []
    out: list[int] = []
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if list(haystack[i:i + n]) == list(needle):
            out.append(i)
    return out


def visible_token(text: str) -> str:
    return text.replace(" ", "·")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rounded(value: Any, digits: int = 3) -> float | str | None:
    f = as_float(value)
    if f is None or not math.isfinite(f):
        return "" if value == "" else None
    return round(f, digits)


def mean(values: Sequence[Any]) -> float | None:
    xs = [as_float(v) for v in values]
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return None
    return sum(xs) / len(xs)


def median(values: Sequence[Any]) -> float | None:
    xs = [as_float(v) for v in values]
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return None
    return float(statistics.median(xs))


def fraction(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return sum(1 for r in rows if bool(r.get(key))) / len(rows)


def row_metric(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in row:
            val = as_float(row.get(key))
            if val is not None:
                return val
    return None


def layer_candidates(n_layers: int, fracs: Sequence[float]) -> list[int]:
    vals = {max(1, min(n_layers, int(round(n_layers * f)))) for f in fracs}
    vals.add(max(1, n_layers // 2))
    vals.add(n_layers)
    return sorted(vals)


def unit_rows(x: Any) -> Any:
    import torch

    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def roc_auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Pairwise AUC.  Returns 0.5 when either class is absent."""
    if not pos or not neg:
        return 0.5
    wins = ties = 0.0
    for p in pos:
        for q in neg:
            wins += float(p > q)
            ties += float(p == q)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def binomial_se(rate: float, n: int) -> float:
    return math.sqrt(max(0.0, rate * (1.0 - rate)) / max(1, n))


def auc_se_hanley_mcneil(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil (1982) standard error of an ROC AUC."""
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1.0 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return math.sqrt(max(0.0, var))


def class_counts_ok(y: Sequence[int | float]) -> bool:
    vals = {int(v) for v in y}
    return vals == {0, 1}


# ---------------------------------------------------------------------------
# Ledger reconciliation
# ---------------------------------------------------------------------------


def parse_ledger() -> list[dict[str, str]]:
    """Parse the two-line claim format produced by the shared bench."""
    if not bench.LEDGER_PATH.exists():
        return []
    entries: list[dict[str, str]] = []
    lines = bench.LEDGER_PATH.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        m = LEDGER_ENTRY_RE.match(line.strip())
        if not m:
            continue
        artifact = falsifier = ""
        if i + 1 < len(lines) and lines[i + 1].startswith("Artifact:"):
            tail = lines[i + 1][len("Artifact:"):]
            artifact, _, falsifier = tail.partition("| Falsifier:")
        entries.append(
            {
                "id": m["id"],
                "tag": m["tag"],
                "text": m["text"].strip(),
                "artifact": artifact.strip(),
                "falsifier": falsifier.strip(),
            }
        )
    return entries


def write_ledger_reconciliation(
    ctx: bench.RunContext,
    entries: list[dict[str, str]],
    domain: str,
    touched_labs: tuple[str, ...],
) -> int:
    touched = [e for e in entries if any(e["id"].startswith(lab) for lab in touched_labs)]
    others = [e for e in entries if e not in touched]

    matrix_rows = []
    for e in entries:
        matrix_rows.append(
            {
                "claim_id": e["id"],
                "original_tag": e["tag"],
                "touched_by_this_audit": any(e["id"].startswith(l) for l in touched_labs),
                "artifact_recorded_in_ledger": e.get("artifact", ""),
                "falsifier_recorded_in_ledger": e.get("falsifier", ""),
                "student_verdict_keep_revise_retire": "",
                "student_reason_artifact_from_this_run": "",
            }
        )
    bench.write_csv_with_context(ctx, ctx.path("tables", "ledger_reconciliation_matrix.csv"), matrix_rows)
    ctx.register_artifact(
        ctx.path("tables", "ledger_reconciliation_matrix.csv"),
        "table",
        "Machine-readable keep/revise/retire worksheet for every parsed ledger claim.",
    )

    lines = [
        "# Ledger reconciliation",
        "",
        "Every claim in your ledger gets a verdict: **keep**, **revise**, or **retire**.",
        "A revision is a narrower claim; a retirement is a finding, not a failure.",
        "At least one claim must be revised or retired.  A semester of perfect claims",
        "usually means the falsifier columns were decorative rather than operational.",
        "",
        "Use `tables/ledger_reconciliation_matrix.csv` as the spreadsheet version of this worksheet.",
        "",
    ]
    if not entries:
        lines += [
            "## Ledger status",
            "",
            "**No claim-ledger entries were parsed.** The capstone audits the ledger;",
            "without entries there is nothing to reconcile.  Each run wrote a",
            "`ledger_suggestions.md`; edit the claims you would defend, append them to",
            "`claim_ledger.md`, and rerun this audit.  The empty-ledger warning is itself",
            "a useful diagnostic: the capstone cannot be assembled backwards.  A semester",
            "with no prior claims to keep/revise/retire usually means the falsifier",
            "columns in earlier labs were never treated as operational.",
            "",
        ]
    if touched:
        lines += [f"## Claims this audit touches directly: `{domain}`", ""]
        for e in touched:
            lines += [
                f"### {e['id']} ({e['tag']})",
                "",
                f"> {e['text']}",
                "",
                f"- recorded artifact: {e['artifact'] or '(none recorded)'}",
                f"- recorded falsifier: {e['falsifier'] or '(none recorded)'}",
                "- **verdict [STUDENT — graded]:** keep / revise / retire",
                "- **reason [STUDENT — graded]:** cite a *specific* artifact + metric from *THIS* run (e.g., 'subject_early recovery 0.995 vs unrelated_clean 0.03 in causal_subset.csv') and say what it did to the original claim",
                "- **replacement claim if revised [STUDENT — graded]:** narrower population / metric / method that survives the counterevidence",
                "",
            ]
    if others:
        lines += ["## Remaining claims", ""]
        for e in others:
            text = e["text"][:140] + ("…" if len(e["text"]) > 140 else "")
            lines += [
                f"- `{e['id']}` {e['tag']}: {text}",
                "  - verdict [STUDENT]: keep / revise / retire",
                "  - reason [STUDENT]:",
                "",
            ]
    lines += [
        "## Minimum standard",
        "",
        "A verdict without an artifact is a mood.  A verdict with an artifact, a metric,",
        "and a scope boundary is an audit.",
    ]
    bench.write_text(ctx.path("ledger_reconciliation.md"), "\n".join(lines))
    ctx.register_artifact(
        ctx.path("ledger_reconciliation.md"),
        "summary",
        "Per-claim keep/revise/retire worksheet.  Verdicts are graded coursework.",
    )
    return len(entries)


# ---------------------------------------------------------------------------
# Residual cache for already-rendered chat prompts
# ---------------------------------------------------------------------------


def run_with_residual_cache_rendered(
    bundle: bench.ModelBundle,
    prompt: str,
    *,
    add_special_tokens: bool,
) -> Any:
    """Residual-cache helper with explicit add_special_tokens control.

    The shared bench intentionally tokenizes ordinary lab prompts itself.  The
    CoT audit, however, receives already-rendered chat prompts from Lab 10.
    Re-adding a BOS/chat wrapper can move the answer-emission position and
    corrupt the probe.  This helper mirrors the bench's residual semantics
    while allowing ``add_special_tokens=False``.
    """
    import torch

    tokenizer = bundle.tokenizer
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    input_ids = encoded["input_ids"].to(bundle.input_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.input_device)

    captured: dict[str, Any] = {}

    def final_norm_pre_hook(module: Any, hook_args: tuple) -> None:
        captured["final_prenorm"] = bench.tensor_cpu_float(hook_args[0])

    handle = bundle.final_norm.register_forward_pre_hook(final_norm_pre_hook)
    try:
        with torch.no_grad():
            out = bundle.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
    finally:
        handle.remove()

    if "final_prenorm" not in captured:
        raise RuntimeError("final-norm pre-hook did not fire for rendered prompt residual capture")
    hs = out.hidden_states
    n_layers = bundle.anatomy.n_layers
    if len(hs) != n_layers + 1:
        raise RuntimeError(f"expected {n_layers + 1} hidden states, got {len(hs)}")

    streams = torch.stack(
        [bench.tensor_cpu_float(h[0]) for h in hs[:-1]] + [captured["final_prenorm"][0]]
    )
    ids = input_ids[0].detach().cpu().tolist()
    raw = tokenizer.convert_ids_to_tokens(ids)
    text = [tokenizer.decode([i]) for i in ids]
    return bench.ForwardCapture(
        prompt=prompt,
        input_ids=ids,
        tokens_raw=raw,
        tokens_text=text,
        streams=streams,
        final_logits_last=bench.tensor_cpu_float(out.logits[0, -1]),
    )


# ---------------------------------------------------------------------------
# Factual QA: data construction and internal evidence
# ---------------------------------------------------------------------------


def resolve_factual_budget(args: Any) -> int:
    max_examples = int(arg_value(args, "max_examples", -1))
    if max_examples > 0:
        return max_examples
    if max_examples == 0:
        return len(FACT_POOL)
    return FACTUAL_BUDGET_BY_TIER.get(str(arg_value(args, "tier", "b")), 12)


def corrupt_partner_for(fid: str) -> tuple[str, str, str]:
    idx = next(i for i, row in enumerate(FACT_POOL) if row[0] == fid)
    current_capital = FACT_POOL[idx][2]
    for offset in range(1, len(FACT_POOL)):
        partner = FACT_POOL[(idx + offset) % len(FACT_POOL)]
        if partner[2] != current_capital:
            return partner
    raise RuntimeError("FACT_POOL has no valid distractor partner")


def build_factual_examples(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    max_facts: int,
) -> tuple[list[AuditExample], list[dict[str, Any]]]:
    tokenizer = bundle.tokenizer
    selected = FACT_POOL[:max(0, min(max_facts, len(FACT_POOL)))] if max_facts > 0 else FACT_POOL
    kept: list[AuditExample] = []
    token_rows: list[dict[str, Any]] = []

    for fid, subject, capital in selected:
        partner_fid, _, partner_capital = corrupt_partner_for(fid)
        subject_text = " " + subject
        target_text = " " + capital
        distractor_text = " " + partner_capital
        subject_ids = token_ids(tokenizer, subject_text)
        target_ids = token_ids(tokenizer, target_text)
        distractor_ids = token_ids(tokenizer, distractor_text)

        for template_id, template in TEMPLATES.items():
            prompt = template.format(X=subject)
            prompt_ids = token_ids(tokenizer, prompt)
            positions = find_subsequence(prompt_ids, subject_ids)
            reasons: list[str] = []
            if len(subject_ids) != 1:
                reasons.append(f"subject tokenizes to {len(subject_ids)} tokens")
            if len(target_ids) != 1:
                reasons.append(f"target tokenizes to {len(target_ids)} tokens")
            if len(distractor_ids) != 1:
                reasons.append(f"distractor tokenizes to {len(distractor_ids)} tokens")
            if len(target_ids) == 1 and len(distractor_ids) == 1 and target_ids[0] == distractor_ids[0]:
                reasons.append("target and distractor have the same token id")
            if len(subject_ids) == 1 and not positions:
                reasons.append("subject token not found in prompt tokenization")
            kept_flag = not reasons
            token_rows.append(
                {
                    "fact_id": fid,
                    "template_id": template_id,
                    "kept": kept_flag,
                    "drop_reason": "; ".join(reasons),
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "n_prompt_tokens": len(prompt_ids),
                    "prompt_token_ids": " ".join(str(i) for i in prompt_ids),
                    "prompt_pieces": token_pieces(tokenizer, prompt_ids),
                    "subject": visible_token(subject_text),
                    "subject_n_tokens": len(subject_ids),
                    "subject_id": subject_ids[0] if len(subject_ids) == 1 else "",
                    "subject_positions": "|".join(str(p) for p in positions),
                    "target": visible_token(target_text),
                    "target_n_tokens": len(target_ids),
                    "target_id": target_ids[0] if len(target_ids) == 1 else "",
                    "target_pieces": token_pieces(tokenizer, target_ids),
                    "distractor": visible_token(distractor_text),
                    "distractor_fact_id": partner_fid,
                    "distractor_n_tokens": len(distractor_ids),
                    "distractor_id": distractor_ids[0] if len(distractor_ids) == 1 else "",
                    "distractor_pieces": token_pieces(tokenizer, distractor_ids),
                }
            )
            if kept_flag:
                kept.append(
                    AuditExample(
                        fact_id=fid,
                        template_id=template_id,
                        prompt=prompt,
                        subject=subject,
                        target=target_text,
                        distractor=distractor_text,
                        target_id=target_ids[0],
                        distractor_id=distractor_ids[0],
                        subject_id=subject_ids[0],
                        subject_pos=positions[-1],
                        n_prompt_tokens=len(prompt_ids),
                        corrupt_fact_id=partner_fid,
                    )
                )

    bench.write_csv_with_context(ctx, ctx.path("diagnostics", "factual_tokenization_report.csv"), token_rows)
    ctx.register_artifact(
        ctx.path("diagnostics", "factual_tokenization_report.csv"),
        "diagnostic",
        "Single-token answer and subject-position gate for factual QA audit examples.",
    )
    return kept, token_rows


def norm_scoring_vector(bundle: bench.ModelBundle, final_stream: Any, target_id: int, distractor_id: int) -> dict[str, Any]:
    """Frozen-final-norm DLA scoring vector, copied in miniature from Lab 2.

    Returns a vector ``w`` such that component_score = component @ w under the
    frozen-norm convention.  RMSNorm and centered LayerNorm are handled
    separately.  Constant terms are reported but not assigned to components.
    """
    import torch

    w_u = bundle.lm_head.weight
    direction = (w_u[target_id].detach() - w_u[distractor_id].detach()).to("cpu", torch.float32)
    norm = bundle.final_norm
    norm_class = type(norm).__name__
    is_rms = "rms" in norm_class.lower()
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5)))
    gain = getattr(norm, "weight", None)
    if gain is None:
        gain_vec = torch.ones_like(direction)
    else:
        gain_vec = gain.detach().to("cpu", torch.float32)

    constant = 0.0
    if is_rms:
        frozen_scale = 1.0 / float(torch.sqrt(final_stream.pow(2).mean() + eps))
        w = frozen_scale * gain_vec * direction
        norm_kind = "rmsnorm"
    else:
        frozen_scale = 1.0 / float(torch.sqrt(final_stream.var(unbiased=False) + eps))
        v = frozen_scale * gain_vec * direction
        w = v - v.mean()
        norm_bias = getattr(norm, "bias", None)
        if norm_bias is not None:
            constant += float(norm_bias.detach().to("cpu", torch.float32) @ direction)
        norm_kind = "layernorm"

    lm_bias = getattr(bundle.lm_head, "bias", None)
    if lm_bias is not None:
        lm_bias = lm_bias.detach().to("cpu", torch.float32)
        constant += float(lm_bias[target_id] - lm_bias[distractor_id])
    return {
        "w": w,
        "constant": constant,
        "norm_class": norm_class,
        "norm_kind": norm_kind,
        "frozen_scale": frozen_scale,
        "answer_direction_norm": float(direction.norm()),
        "scoring_vector_norm": float(w.norm()),
    }


def dla_layer_summary(bundle: bench.ModelBundle, comp: Any, target_id: int, distractor_id: int) -> dict[str, Any]:
    import torch

    x_final = comp.capture.streams[-1, -1]
    s = norm_scoring_vector(bundle, x_final, target_id, distractor_id)
    w = s["w"]
    per_layer: list[dict[str, Any]] = []
    for layer in range(bundle.anatomy.n_layers):
        attn = float(comp.attn_contrib[layer] @ w)
        mlp = float(comp.mlp_contrib[layer] @ w)
        per_layer.append(
            {
                "layer": layer,
                "attn": attn,
                "mlp": mlp,
                "block_total": attn + mlp,
                "abs_block_total": abs(attn) + abs(mlp),
            }
        )
    embed = float(comp.capture.streams[0, -1] @ w)
    attn_total = sum(r["attn"] for r in per_layer)
    mlp_total = sum(r["mlp"] for r in per_layer)
    ledger_total = embed + attn_total + mlp_total + float(s["constant"])
    model_logit_diff = float(comp.capture.final_logits_last[target_id] - comp.capture.final_logits_last[distractor_id])
    frozen_logit_diff = float(x_final @ w) + float(s["constant"])
    best = max(per_layer, key=lambda r: r["abs_block_total"])
    best_kind = "mlp" if abs(best["mlp"]) >= abs(best["attn"]) else "attn"
    return {
        "per_layer": per_layer,
        "top_layer": int(best["layer"]),
        "top_kind": best_kind,
        "embed_score": embed,
        "attn_total": attn_total,
        "mlp_total": mlp_total,
        "constant": float(s["constant"]),
        "ledger_total": ledger_total,
        "frozen_logit_diff": frozen_logit_diff,
        "model_logit_diff": model_logit_diff,
        "balance_error": ledger_total - model_logit_diff,
        "frozen_gap": frozen_logit_diff - model_logit_diff,
        "norm_kind": s["norm_kind"],
        "norm_class": s["norm_class"],
        "frozen_scale": s["frozen_scale"],
        "answer_direction_norm": s["answer_direction_norm"],
        "scoring_vector_norm": s["scoring_vector_norm"],
    }


def stabilization_depth(traj: Any) -> int | None:
    """First depth where target is top-1 and stays top-1 to the end."""
    if traj.target_rank is None:
        return None
    candidate: int | None = None
    for depth in range(traj.n_depths):
        if traj.target_rank[depth] == 1:
            if candidate is None:
                candidate = depth
        else:
            candidate = None
    return candidate


def preference_depth(traj: Any) -> int | None:
    """First depth where target beats distractor and keeps doing so."""
    if traj.logit_target is None or traj.logit_distractor is None:
        return None
    candidate: int | None = None
    for depth in range(traj.n_depths):
        if traj.logit_target[depth] > traj.logit_distractor[depth]:
            if candidate is None:
                candidate = depth
        else:
            candidate = None
    return candidate


def auto_failure_label(answer: str, correct: bool, prefers_target: bool, top1_is_distractor: bool) -> str:
    stripped = answer.strip()
    if correct:
        return "correct_exact"
    if prefers_target:
        return "format_or_alias_target_preferred"
    if top1_is_distractor:
        return "distractor_win"
    if stripped.istitle() and len(stripped) > 2:
        return "other_entity"
    if not stripped:
        return "empty_or_special_token"
    return "non_answer_token"


def write_factual_domain_manifest(ctx: bench.RunContext, bundle: bench.ModelBundle, examples: list[AuditExample], token_rows: list[dict[str, Any]]) -> None:
    manifest = {
        "domain": "factual_qa",
        "model_id": bundle.anatomy.model_id,
        "n_layers": bundle.anatomy.n_layers,
        "d_model": bundle.anatomy.d_model,
        "templates": TEMPLATES,
        "n_examples_kept": len(examples),
        "n_facts_kept": len({e.fact_id for e in examples}),
        "n_tokenization_rows": len(token_rows),
        "stream_convention": "bench streams[k]: pre-norm residual after k blocks; streams[0] is embeddings; streams[L] is final-norm input",
        "behavior_metric": "next-token target vs cyclic-partner distractor logit difference",
        "causal_sites": ["subject_early", "final_band", "unrelated_clean_control at both sites"],
        "truth_monitor": "true/false capital statements, held-out facts, shuffled-label control",
    }
    bench.write_json(ctx.path("diagnostics", "audit_domain_manifest.json"), manifest)
    ctx.register_artifact(ctx.path("diagnostics", "audit_domain_manifest.json"), "diagnostic", "Factual audit design, metric, and stream conventions.")


def run_factual_audit(ctx: bench.RunContext, bundle: bench.ModelBundle, args: Any) -> dict[str, Any]:
    import torch

    max_facts = resolve_factual_budget(args)
    examples, token_rows = build_factual_examples(ctx, bundle, max_facts)
    if not examples:
        raise RuntimeError("No factual audit examples survived tokenization. See diagnostics/factual_tokenization_report.csv")
    print(
        f"[lab11] factual_qa: {len(examples)} examples "
        f"({len({e.fact_id for e in examples})} facts x {len(TEMPLATES)} templates after tokenization)"
    )
    write_factual_domain_manifest(ctx, bundle, examples, token_rows)

    comp_anatomy = bench.resolve_component_anatomy(ctx, bundle, examples[0].prompt)
    first_comp = bench.run_with_component_cache(bundle, examples[0].prompt, comp_anatomy)
    bench.run_decomposition_check(
        ctx,
        bundle,
        first_comp,
        rel_tolerance=float(arg_value(args, "dla_tolerance", 0.02)),
    )
    bench.run_patch_noop_check(ctx, bundle, examples[0].prompt)

    rows: list[dict[str, Any]] = []
    lens_rows: list[dict[str, Any]] = []
    dla_rows: list[dict[str, Any]] = []
    captures: dict[tuple[str, str], Any] = {}

    for idx, ex in enumerate(examples):
        comp = first_comp if idx == 0 else bench.run_with_component_cache(bundle, ex.prompt, comp_anatomy)
        captures[(ex.fact_id, ex.template_id)] = comp
        traj = bench.compute_lens_trajectory(
            bundle,
            comp.capture,
            target_id=ex.target_id,
            distractor_id=ex.distractor_id,
            topk=int(arg_value(args, "topk", 5)),
        )
        dla = dla_layer_summary(bundle, comp, ex.target_id, ex.distractor_id)
        final = comp.capture.final_logits_last
        probs = torch.softmax(final, dim=-1)
        top = torch.topk(final, k=5)
        top1 = int(top.indices[0])
        answer = bundle.tokenizer.decode([top1])
        correct = top1 == ex.target_id
        logit_diff = float(final[ex.target_id] - final[ex.distractor_id])
        prefers = logit_diff > 0
        target_rank = traj.target_rank[-1] if traj.target_rank is not None else ""
        distractor_rank = traj.distractor_rank[-1] if traj.distractor_rank is not None else ""
        pref_depth = preference_depth(traj)
        top1_depth = stabilization_depth(traj)
        top1_is_distractor = top1 == ex.distractor_id
        row = {
            "fact_id": ex.fact_id,
            "template_id": ex.template_id,
            "prompt": ex.prompt,
            "prompt_sha256": sha256_text(ex.prompt),
            "subject_pos": ex.subject_pos,
            "target": ex.target,
            "distractor": ex.distractor,
            "corrupt_fact_id": ex.corrupt_fact_id,
            "answer_top1": answer,
            "top1_token_id": top1,
            "target_token_id": ex.target_id,
            "distractor_token_id": ex.distractor_id,
            "correct_exact_top1": correct,
            "prefers_target_over_distractor": prefers,
            "p_target": round(float(probs[ex.target_id]), 6),
            "p_distractor": round(float(probs[ex.distractor_id]), 6),
            "logit_diff_target_minus_distractor": round(logit_diff, 4),
            "confidence_margin_top1_minus_top2": round(float(top.values[0] - top.values[1]), 4),
            "target_rank_final": target_rank,
            "distractor_rank_final": distractor_rank,
            "lens_top1_stabilization_depth": top1_depth if top1_depth is not None else "",
            "lens_preference_stabilization_depth": pref_depth if pref_depth is not None else "",
            "lens_preference_depth_frac": round(pref_depth / bundle.anatomy.n_layers, 3) if pref_depth is not None else "",
            "dla_top_layer": dla["top_layer"],
            "dla_top_component_type": dla["top_kind"],
            "dla_embed_score": round(dla["embed_score"], 3),
            "dla_attn_total": round(dla["attn_total"], 3),
            "dla_mlp_total": round(dla["mlp_total"], 3),
            "dla_constant": round(dla["constant"], 3),
            "dla_ledger_total": round(dla["ledger_total"], 3),
            "dla_model_logit_diff": round(dla["model_logit_diff"], 3),
            "dla_balance_error": round(dla["balance_error"], 4),
            "failure_mode_auto": auto_failure_label(answer, correct, prefers, top1_is_distractor),
            "failure_mode_student": "",
        }
        rows.append(row)
        lens_rows.append(
            {
                "fact_id": ex.fact_id,
                "template_id": ex.template_id,
                "top1_stabilization_depth": top1_depth if top1_depth is not None else "",
                "preference_stabilization_depth": pref_depth if pref_depth is not None else "",
                "final_target_rank": target_rank,
                "final_distractor_rank": distractor_rank,
                "final_entropy_bits": round(traj.entropy_bits[-1], 3),
                "final_kl_to_final_bits": round(traj.kl_to_final_bits[-1], 6),
                "first_depth_target_top1": top1_depth if top1_depth is not None else "not stable",
                "first_depth_target_prefers_over_distractor": pref_depth if pref_depth is not None else "not stable",
            }
        )
        for r in dla["per_layer"]:
            dla_rows.append(
                {
                    "fact_id": ex.fact_id,
                    "template_id": ex.template_id,
                    "layer": r["layer"],
                    "attn_score": round(r["attn"], 4),
                    "mlp_score": round(r["mlp"], 4),
                    "block_total": round(r["block_total"], 4),
                    "abs_block_total": round(r["abs_block_total"], 4),
                    "model_logit_diff": round(dla["model_logit_diff"], 4),
                    "norm_kind": dla["norm_kind"],
                    "frozen_scale": round(dla["frozen_scale"], 6),
                }
            )

    bench.write_csv_with_context(ctx, ctx.path("results.csv"), rows)
    ctx.register_artifact(ctx.path("results.csv"), "results", "Per example audit table: behavior, confidence, lens, DLA, and failure-mode labels.")
    bench.write_csv_with_context(ctx, ctx.path("tables", "lens_stabilization.csv"), lens_rows)
    ctx.register_artifact(ctx.path("tables", "lens_stabilization.csv"), "table", "Per-example logit-lens stabilization depths and final ranks.")
    bench.write_csv_with_context(ctx, ctx.path("tables", "dla_layer_summary.csv"), dla_rows)
    ctx.register_artifact(ctx.path("tables", "dla_layer_summary.csv"), "table", "Per-example, per-layer DLA scores under frozen final norm.")

    fact_rows = summarize_paraphrases(ctx, rows)
    depth_values = [r["lens_preference_stabilization_depth"] for r in rows if r["lens_preference_stabilization_depth"] != ""]
    global_band = int(round(median(depth_values) or max(1, bundle.anatomy.n_layers // 2)))
    global_band = max(1, min(global_band, bundle.anatomy.n_layers))

    causal_rows, causal_candidates = run_factual_causal_subset(
        ctx,
        bundle,
        examples,
        rows,
        captures,
        global_band,
    )
    monitor = run_truth_monitor(ctx, bundle, examples)
    maybe_make_factual_plots(ctx, bundle, rows, fact_rows, causal_rows, monitor)
    write_evidence_matrix(ctx, "factual_qa", rows, causal_rows, monitor)

    target_patch_rows = [c for c in causal_rows if c.get("condition") == "target_clean_patch"]
    behavioral = {
        "n_examples": len(rows),
        "n_facts": len({r["fact_id"] for r in rows}),
        "n_tokenization_dropped": sum(1 for r in token_rows if not r["kept"]),
        "top1_exact_accuracy": round(fraction(rows, "correct_exact_top1") or 0.0, 3),
        "target_preference_accuracy": round(fraction(rows, "prefers_target_over_distractor") or 0.0, 3),
        "paraphrase_consistent_facts": round(fraction(fact_rows, "consistent_preference") or 0.0, 3),
        "median_preference_depth": global_band,
        "median_preference_depth_frac": round(global_band / bundle.anatomy.n_layers, 3),
        "mean_recovery_subject_early": rounded(mean([c["recovery"] for c in target_patch_rows if c["site"] == "subject_early"]), 3),
        "mean_recovery_final_band": rounded(mean([c["recovery"] for c in target_patch_rows if c["site"] == "final_band"]), 3),
        "mean_recovery_unrelated_control": rounded(mean([c["recovery"] for c in causal_rows if c.get("condition") == "unrelated_clean_control"]), 3),
        "n_causal_target_patches": len(target_patch_rows),
        "n_causal_candidates": len(causal_candidates),
        "monitor": monitor,
    }
    return {
        "rows": rows,
        "fact_rows": fact_rows,
        "causal_rows": causal_rows,
        "causal_candidates": causal_candidates,
        "behavioral": behavioral,
        "touched_labs": ("L01", "L02", "L04", "L05"),
    }


def summarize_paraphrases(ctx: bench.RunContext, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_fact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_fact[str(row["fact_id"])].append(row)
    fact_rows: list[dict[str, Any]] = []
    for fid, rs in sorted(by_fact.items()):
        pref_depths = [r["lens_preference_stabilization_depth"] for r in rs if r["lens_preference_stabilization_depth"] != ""]
        logit_diffs = [float(r["logit_diff_target_minus_distractor"]) for r in rs]
        fact_rows.append(
            {
                "fact_id": fid,
                "n_templates": len(rs),
                "n_correct_exact_top1": sum(bool(r["correct_exact_top1"]) for r in rs),
                "n_prefers_target": sum(bool(r["prefers_target_over_distractor"]) for r in rs),
                "consistent_top1_text": len({r["answer_top1"] for r in rs}) == 1,
                "consistent_preference": all(bool(r["prefers_target_over_distractor"]) for r in rs),
                "min_logit_diff": round(min(logit_diffs), 4),
                "max_logit_diff": round(max(logit_diffs), 4),
                "mean_logit_diff": round(sum(logit_diffs) / len(logit_diffs), 4),
                "preference_depth_min": min(pref_depths) if pref_depths else "",
                "preference_depth_max": max(pref_depths) if pref_depths else "",
                "preference_depth_spread": (max(pref_depths) - min(pref_depths)) if pref_depths else "",
                "worst_template_by_logit_diff": min(rs, key=lambda r: float(r["logit_diff_target_minus_distractor"]))["template_id"],
            }
        )
    bench.write_csv_with_context(ctx, ctx.path("tables", "paraphrase_consistency.csv"), fact_rows)
    ctx.register_artifact(ctx.path("tables", "paraphrase_consistency.csv"), "table", "Per-fact consistency across the three factual templates.")
    return fact_rows


def run_factual_causal_subset(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    examples: list[AuditExample],
    rows: list[dict[str, Any]],
    captures: dict[tuple[str, str], Any],
    global_band: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    row_by_key = {(r["fact_id"], r["template_id"]): r for r in rows}
    base_examples = [e for e in examples if e.template_id == "base"]
    base_by_fact = {e.fact_id: e for e in base_examples}
    candidates: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []

    for ex in base_examples:
        clean_row = row_by_key[(ex.fact_id, "base")]
        partner = base_by_fact.get(ex.corrupt_fact_id)
        reason: list[str] = []
        if partner is None:
            reason.append("corrupt partner not in selected/kept facts")
        if not bool(clean_row["prefers_target_over_distractor"]):
            reason.append("clean prompt does not prefer target over distractor")
        corrupt_base_diff = ""
        corrupt_prefers_distractor = False
        if partner is not None:
            corrupt_logits = bench.next_token_logits(bundle, partner.prompt)
            corrupt_base_diff_f = float(corrupt_logits[ex.target_id] - corrupt_logits[ex.distractor_id])
            corrupt_base_diff = round(corrupt_base_diff_f, 4)
            corrupt_prefers_distractor = corrupt_base_diff_f < 0
            if not corrupt_prefers_distractor:
                reason.append("corrupt prompt does not prefer the clean example's distractor")
        chosen = not reason
        candidates.append(
            {
                "fact_id": ex.fact_id,
                "corrupt_fact_id": ex.corrupt_fact_id,
                "chosen_for_causal_subset": chosen,
                "reason_if_not_chosen": "; ".join(reason),
                "clean_logit_diff": clean_row["logit_diff_target_minus_distractor"],
                "corrupt_logit_diff_on_clean_metric": corrupt_base_diff,
                "subject_pos": ex.subject_pos,
                "example_preference_depth": clean_row["lens_preference_stabilization_depth"],
            }
        )

    chosen_examples = [base_by_fact[c["fact_id"]] for c in candidates if c["chosen_for_causal_subset"]]
    chosen_examples = chosen_examples[:N_CAUSAL_SUBSET]
    unrelated_pool = [e for e in base_examples if e.fact_id not in {x.fact_id for x in chosen_examples}]

    for ex in chosen_examples:
        partner = base_by_fact[ex.corrupt_fact_id]
        clean = captures[(ex.fact_id, "base")].capture
        corrupt_prompt = partner.prompt
        corrupt_logits = bench.next_token_logits(bundle, corrupt_prompt)
        corrupt_diff = float(corrupt_logits[ex.target_id] - corrupt_logits[ex.distractor_id])
        clean_row = row_by_key[(ex.fact_id, "base")]
        clean_diff = float(clean_row["logit_diff_target_minus_distractor"])
        denom = clean_diff - corrupt_diff
        if abs(denom) < 1e-6:
            continue
        per_depth = clean_row["lens_preference_stabilization_depth"]
        final_depth = int(per_depth) if per_depth != "" else global_band
        final_depth = max(1, min(final_depth, bundle.anatomy.n_layers))
        early_depth = max(1, min(bundle.anatomy.n_layers, final_depth // 2))
        unrelated = next((u for u in unrelated_pool if u.fact_id not in {ex.fact_id, partner.fact_id}), None)
        if unrelated is None:
            unrelated = next((u for u in base_examples if u.fact_id not in {ex.fact_id, partner.fact_id}), None)
        sites = (
            ("subject_early", early_depth, ex.subject_pos),
            ("final_band", final_depth, clean.streams.shape[1] - 1),
        )
        for site, depth, pos in sites:
            target_patch = clean.streams[depth, pos]
            patched_logits = bench.run_with_residual_patch(bundle, corrupt_prompt, depth, pos, target_patch)
            patched_diff = float(patched_logits[ex.target_id] - patched_logits[ex.distractor_id])
            causal_rows.append(
                {
                    "clean_fact": ex.fact_id,
                    "corrupt_fact": partner.fact_id,
                    "condition": "target_clean_patch",
                    "site": site,
                    "stream_depth": depth,
                    "patch_pos": pos,
                    "clean_logit_diff": round(clean_diff, 4),
                    "corrupt_logit_diff": round(corrupt_diff, 4),
                    "patched_logit_diff": round(patched_diff, 4),
                    "recovery": round((patched_diff - corrupt_diff) / denom, 4),
                    "denominator_clean_minus_corrupt": round(denom, 4),
                    "control_source_fact": "",
                }
            )
            if unrelated is not None:
                unrelated_cap = captures[(unrelated.fact_id, "base")].capture
                if pos < unrelated_cap.streams.shape[1]:
                    control_vector = unrelated_cap.streams[depth, pos]
                    control_logits = bench.run_with_residual_patch(bundle, corrupt_prompt, depth, pos, control_vector)
                    control_diff = float(control_logits[ex.target_id] - control_logits[ex.distractor_id])
                    causal_rows.append(
                        {
                            "clean_fact": ex.fact_id,
                            "corrupt_fact": partner.fact_id,
                            "condition": "unrelated_clean_control",
                            "site": site,
                            "stream_depth": depth,
                            "patch_pos": pos,
                            "clean_logit_diff": round(clean_diff, 4),
                            "corrupt_logit_diff": round(corrupt_diff, 4),
                            "patched_logit_diff": round(control_diff, 4),
                            "recovery": round((control_diff - corrupt_diff) / denom, 4),
                            "denominator_clean_minus_corrupt": round(denom, 4),
                            "control_source_fact": unrelated.fact_id,
                        }
                    )

    bench.write_csv_with_context(ctx, ctx.path("tables", "causal_candidate_manifest.csv"), candidates)
    ctx.register_artifact(ctx.path("tables", "causal_candidate_manifest.csv"), "table", "Why each base fact was or was not eligible for the causal patch subset.")
    bench.write_csv_with_context(ctx, ctx.path("tables", "causal_subset.csv"), causal_rows)
    ctx.register_artifact(ctx.path("tables", "causal_subset.csv"), "table", "Residual patches at two stream sites plus unrelated-clean controls.")
    return causal_rows, candidates


# ---------------------------------------------------------------------------
# Factual QA: truth monitor
# ---------------------------------------------------------------------------


def latest_compatible_truth_direction(bundle: bench.ModelBundle) -> dict[str, Any] | None:
    """Load the latest Lab 4 direction only when it is model-compatible."""
    import torch

    run_root = bench.COURSE_ROOT / "runs"
    if not run_root.exists():
        return None
    saved = sorted(run_root.glob("lab04*/tables/truth_direction.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in saved:
        try:
            meta = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        vec = meta.get("direction") if isinstance(meta, dict) else None
        if vec is None:
            continue
        vec_t = torch.as_tensor(vec, dtype=torch.float32)
        if int(vec_t.numel()) != int(bundle.anatomy.d_model):
            continue
        if meta.get("model_id") and str(meta.get("model_id")) != str(bundle.anatomy.model_id):
            continue
        layer = int(meta.get("layer", -1))
        if not 0 <= layer <= bundle.anatomy.n_layers:
            continue
        return {"path": path, "meta": meta, "direction": vec_t, "layer": layer}
    return None


def run_truth_monitor(ctx: bench.RunContext, bundle: bench.ModelBundle, examples: list[AuditExample]) -> dict[str, Any]:
    import torch

    base_examples = [e for e in examples if e.template_id == "base"]
    statements: list[dict[str, Any]] = []
    for e in base_examples:
        statements.append({"fact_id": e.fact_id, "statement": f"The capital of {e.subject} is{e.target}.", "label": 1, "kind": "true_target"})
        statements.append({"fact_id": e.fact_id, "statement": f"The capital of {e.subject} is{e.distractor}.", "label": 0, "kind": "false_distractor"})
    if len({s["fact_id"] for s in statements}) < 2:
        result = {"status": "skipped", "reason": "need at least two facts for held-out monitor", "held_out_auc": None, "shuffled_control_auc": None, "selectivity": None}
        bench.write_json(ctx.path("internal_evidence", "truth_monitor.json"), result)
        ctx.register_artifact(ctx.path("internal_evidence", "truth_monitor.json"), "metrics", "Truth monitor skipped reason.")
        return result

    unique_facts = sorted({s["fact_id"] for s in statements})
    split_point = max(1, len(unique_facts) // 2)
    train_facts = set(unique_facts[:split_point])
    for s in statements:
        s["split"] = "train" if s["fact_id"] in train_facts else "heldout"
        s["statement_sha256"] = sha256_text(str(s["statement"]))

    candidates = layer_candidates(bundle.anatomy.n_layers, TRUTH_MONITOR_LAYER_FRACS)
    captures = [bench.run_with_residual_cache(bundle, str(s["statement"])) for s in statements]
    labels = [int(s["label"]) for s in statements]
    train_idx = [i for i, s in enumerate(statements) if s["split"] == "train"]
    held_idx = [i for i, s in enumerate(statements) if s["split"] == "heldout"]

    compatible = latest_compatible_truth_direction(bundle)
    sweep_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None

    if compatible is not None:
        layer = compatible["layer"]
        X = unit_rows(torch.stack([cap.streams[layer, -1] for cap in captures]))
        y = torch.tensor(labels, dtype=torch.float32)
        direction = compatible["direction"] / compatible["direction"].norm().clamp_min(1e-9)
        proj = (X @ direction).tolist()
        auc_train = roc_auc([proj[i] for i in train_idx if labels[i] == 1], [proj[i] for i in train_idx if labels[i] == 0])
        auc_held = roc_auc([proj[i] for i in held_idx if labels[i] == 1], [proj[i] for i in held_idx if labels[i] == 0])
        gen = torch.Generator().manual_seed(0)
        y_shuf = y[train_idx][torch.randperm(len(train_idx), generator=gen)]
        ds = X[train_idx][y_shuf == 1].mean(0) - X[train_idx][y_shuf == 0].mean(0)
        ds = ds / ds.norm().clamp_min(1e-9)
        proj_s = (X @ ds).tolist()
        auc_shuf = roc_auc([proj_s[i] for i in held_idx if labels[i] == 1], [proj_s[i] for i in held_idx if labels[i] == 0])
        chosen = {
            "source": f"Lab 4 artifact ({compatible['path'].parent.parent.name})",
            "layer": layer,
            "direction": direction,
            "projection": proj,
            "shuffled_projection": proj_s,
            "train_auc": auc_train,
            "held_out_auc": auc_held,
            "shuffled_control_auc": auc_shuf,
            "selectivity": auc_held - auc_shuf,
            "selection_rule": "latest compatible Lab 4 truth_direction.pt at its saved stream depth; shuffled control refit locally on train facts",
        }
        sweep_rows.append({"layer": layer, "source": chosen["source"], "train_auc": round(auc_train, 3), "held_out_auc": round(auc_held, 3), "shuffled_control_auc": round(auc_shuf, 3), "selectivity": round(auc_held - auc_shuf, 3), "selected": True})
    else:
        gen = torch.Generator().manual_seed(0)
        for layer in candidates:
            X = unit_rows(torch.stack([cap.streams[layer, -1] for cap in captures]))
            y = torch.tensor(labels, dtype=torch.float32)
            if not class_counts_ok([labels[i] for i in train_idx]) or not class_counts_ok([labels[i] for i in held_idx]):
                continue
            d = X[train_idx][y[train_idx] == 1].mean(0) - X[train_idx][y[train_idx] == 0].mean(0)
            d = d / d.norm().clamp_min(1e-9)
            proj = (X @ d).tolist()
            auc_train = roc_auc([proj[i] for i in train_idx if labels[i] == 1], [proj[i] for i in train_idx if labels[i] == 0])
            auc_held = roc_auc([proj[i] for i in held_idx if labels[i] == 1], [proj[i] for i in held_idx if labels[i] == 0])
            y_shuf = y[train_idx][torch.randperm(len(train_idx), generator=gen)]
            ds = X[train_idx][y_shuf == 1].mean(0) - X[train_idx][y_shuf == 0].mean(0)
            ds = ds / ds.norm().clamp_min(1e-9)
            proj_s = (X @ ds).tolist()
            auc_shuf = roc_auc([proj_s[i] for i in held_idx if labels[i] == 1], [proj_s[i] for i in held_idx if labels[i] == 0])
            row = {
                "layer": layer,
                "source": "mass-mean trained on train facts",
                "train_auc": round(auc_train, 3),
                "held_out_auc": round(auc_held, 3),
                "shuffled_control_auc": round(auc_shuf, 3),
                "selectivity": round(auc_held - auc_shuf, 3),
                "selected": False,
            }
            sweep_rows.append(row)
            score = auc_train - 0.5
            if chosen is None or score > chosen["selection_score"]:
                chosen = {
                    "source": "mass-mean (trained on half the audited facts)",
                    "layer": layer,
                    "direction": d,
                    "projection": proj,
                    "shuffled_projection": proj_s,
                    "train_auc": auc_train,
                    "held_out_auc": auc_held,
                    "shuffled_control_auc": auc_shuf,
                    "selectivity": auc_held - auc_shuf,
                    "selection_score": score,
                    "selection_rule": "choose layer by train-fact AUC; report held-out facts vs shuffled control",
                }
        for row in sweep_rows:
            if chosen is not None and int(row["layer"]) == int(chosen["layer"]):
                row["selected"] = True

    if chosen is None:
        result = {"status": "skipped", "reason": "not enough class coverage after split", "held_out_auc": None, "shuffled_control_auc": None, "selectivity": None}
        bench.write_json(ctx.path("internal_evidence", "truth_monitor.json"), result)
        ctx.register_artifact(ctx.path("internal_evidence", "truth_monitor.json"), "metrics", "Truth monitor skipped reason.")
        return result

    proj = chosen["projection"]
    for i, s in enumerate(statements):
        projection_rows.append(
            {
                **s,
                "layer": chosen["layer"],
                "projection": round(float(proj[i]), 5),
                "source": chosen["source"],
            }
        )

    bench.write_csv_with_context(ctx, ctx.path("tables", "truth_monitor_statements.csv"), projection_rows)
    ctx.register_artifact(ctx.path("tables", "truth_monitor_statements.csv"), "table", "Truth-monitor statements, splits, labels, and chosen-direction projections.")
    bench.write_csv_with_context(ctx, ctx.path("tables", "truth_monitor_layer_sweep.csv"), sweep_rows)
    ctx.register_artifact(ctx.path("tables", "truth_monitor_layer_sweep.csv"), "table", "Candidate truth-monitor depths and train/held-out AUCs.")

    result = {
        "status": "ok",
        "source": chosen["source"],
        "layer": int(chosen["layer"]),
        "n_statements": len(statements),
        "n_train_statements": len(train_idx),
        "n_heldout_statements": len(held_idx),
        "train_auc": round(float(chosen["train_auc"]), 3),
        "held_out_auc": round(float(chosen["held_out_auc"]), 3),
        "shuffled_control_auc": rounded(chosen.get("shuffled_control_auc"), 3),
        "selectivity": rounded(chosen.get("selectivity"), 3),
        "selection_rule": chosen["selection_rule"],
        "normalization": "row-unit-normalized residual streams before projection",
    }
    bench.write_json(ctx.path("internal_evidence", "truth_monitor.json"), result)
    ctx.register_artifact(ctx.path("internal_evidence", "truth_monitor.json"), "metrics", "Truth-direction monitor AUC on held-out facts vs shuffled control.")
    print(
        f"[lab11] truth monitor: {result['source']}; held-out AUC {result['held_out_auc']} "
        f"at stream depth {result['layer']}"
    )
    return result


# ---------------------------------------------------------------------------
# CoT faithfulness audit
# ---------------------------------------------------------------------------


def load_fresh_cot_items(ctx: bench.RunContext, lab10: Any, cap: int) -> list[dict[str, str]]:
    """Load a fresh item slice while tolerating original and revised Lab 10 APIs."""
    args = ctx.args
    all_items: list[dict[str, str]]
    source_path = "Lab 10 default"
    source_kind = "legacy_load_items"

    if hasattr(lab10, "read_items_from_path") and hasattr(lab10, "resolve_item_source"):
        source, source_kind, _source_cap = lab10.resolve_item_source(args)
        all_items = lab10.read_items_from_path(source)
        source_path = str(source)
    elif hasattr(lab10, "load_items"):
        all_items = lab10.load_items(0)
    else:
        raise RuntimeError("Lab 10 module exposes neither revised read_items_from_path nor legacy load_items")

    # Stable fresh slice: offset from Lab 10's default selection and stride, then
    # roughly round-robin by domain.  This is intentionally simple and written
    # to a manifest; capstone replication should not hide its sampling rule in a
    # clever iterator.
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in all_items:
        grouped[str(item.get("domain", "unknown"))].append(item)
    for items in grouped.values():
        items.sort(key=lambda x: str(x.get("id", "")))

    selected: list[dict[str, str]] = []
    offsets = {d: COT_FRESH_OFFSET for d in grouped}
    while len(selected) < cap:
        progressed = False
        for domain in sorted(grouped):
            items = grouped[domain]
            idx = offsets[domain]
            if idx < len(items):
                selected.append(items[idx])
                offsets[domain] += COT_FRESH_STRIDE
                progressed = True
                if len(selected) >= cap:
                    break
        if not progressed:
            break

    manifest = {
        "domain": "cot_faithfulness",
        "source_path": source_path,
        "source_kind": source_kind,
        "n_source_items": len(all_items),
        "n_selected_items": len(selected),
        "cap": cap,
        "fresh_slice_offset": COT_FRESH_OFFSET,
        "fresh_slice_stride": COT_FRESH_STRIDE,
        "selection_rule": "domain-round-robin after sorting by id; start at offset 1 and stride by 4 to avoid Lab 10's default slice",
        "domain_counts_source": dict(sorted((d, len(v)) for d, v in grouped.items())),
        "domain_counts_selected": dict(sorted((d, sum(1 for x in selected if str(x.get("domain", "unknown")) == d)) for d in grouped)),
    }
    if isinstance(source_path, str) and pathlib.Path(source_path).exists():
        manifest["source_sha256"] = bench.sha256_file(pathlib.Path(source_path))
    bench.write_json(ctx.path("diagnostics", "cot_fresh_slice_manifest.json"), manifest)
    ctx.register_artifact(ctx.path("diagnostics", "cot_fresh_slice_manifest.json"), "diagnostic", "Fresh-slice item selection rule for the CoT audit.")

    item_rows = []
    wrong_fn = getattr(lab10, "wrong_letter", None)
    for item in selected:
        wrong = wrong_fn(item) if callable(wrong_fn) else ""
        key = str(item.get("answer_key", "")).upper()
        item_rows.append(
            {
                "item_id": item.get("id", ""),
                "domain": item.get("domain", ""),
                "answer_key": key,
                "answer_text": item.get(f"option_{key.lower()}", ""),
                "wrong_hint_letter": wrong,
                "wrong_hint_text": item.get(f"option_{str(wrong).lower()}", "") if wrong else "",
                "question_sha256": sha256_text(str(item.get("question", ""))),
            }
        )
    bench.write_csv_with_context(ctx, ctx.path("tables", "cot_fresh_item_manifest.csv"), item_rows)
    ctx.register_artifact(ctx.path("tables", "cot_fresh_item_manifest.csv"), "table", "Fresh CoT audit items and deterministic wrong hints.")
    return selected


def call_acknowledgment_writer(ctx: bench.RunContext, bundle: bench.ModelBundle, lab10: Any, rows: list[dict[str, Any]]) -> None:
    writer = getattr(lab10, "write_acknowledgment_samples", None)
    if not callable(writer):
        return
    try:
        sig = inspect.signature(writer)
        # Legacy Lab 10: (ctx, bundle, rows). Revised Lab 10: (ctx, rows, ...).
        if "bundle" in sig.parameters:
            writer(ctx, bundle, rows)
        else:
            writer(ctx, rows)
    except TypeError:
        try:
            writer(ctx, rows)
        except TypeError:
            writer(ctx, bundle, rows)


def condition_flipped_to_hint(row: Mapping[str, Any]) -> bool:
    return bool(row.get("flipped_to_wrong_hint", row.get("flipped_to_hint", False)))


def condition_silent_flip(row: Mapping[str, Any]) -> bool:
    return condition_flipped_to_hint(row) and not bool(row.get("auto_mention", False))


def run_cot_audit(ctx: bench.RunContext, bundle: bench.ModelBundle, args: Any) -> dict[str, Any]:
    import torch

    import labs.lab10_cot_faithfulness as lab10

    if not bench.supports_chat_template(bundle):
        raise RuntimeError(
            "The cot_faithfulness audit needs a chat/reasoning model. Pass "
            "--model allenai/Olmo-3-7B-Think for the flagship, or a small Think/chat model for smoke."
        )

    cap = int(arg_value(args, "max_examples", -1))
    if cap <= 0:
        cap = COT_AUDIT_ITEMS_BY_TIER.get(str(arg_value(args, "tier", "b")), 16)
    items = load_fresh_cot_items(ctx, lab10, cap)
    if not items:
        raise RuntimeError("No CoT audit items selected. Check diagnostics/cot_fresh_slice_manifest.json")

    max_new = int(getattr(lab10, "MAX_NEW_BY_TIER", {}).get(arg_value(args, "tier", "b"), 1024))
    batch = int(getattr(lab10, "BATCH_BY_TIER", {}).get(arg_value(args, "tier", "b"), 8))
    print(f"[lab11] cot_faithfulness: {len(items)} fresh items; max_new={max_new}, batch={batch}")

    lab10.run_think_roundtrip_check(ctx, bundle, items, max_new, batch)
    rows = lab10.run_hint_experiment(ctx, bundle, items, max_new=max_new, batch=batch)
    public = [{k: v for k, v in r.items() if not str(k).startswith("_")} for r in rows]
    for r in public:
        r.setdefault("failure_mode_auto", cot_failure_auto_label(r))
        r.setdefault("failure_mode_student", "")
    bench.write_csv_with_context(ctx, ctx.path("results.csv"), public)
    ctx.register_artifact(ctx.path("results.csv"), "results", "Per item x condition on the fresh CoT audit slice.")

    table = lab10.faithfulness_table(rows)
    bench.write_csv_with_context(ctx, ctx.path("tables", "faithfulness_by_hint_type.csv"), table)
    ctx.register_artifact(ctx.path("tables", "faithfulness_by_hint_type.csv"), "table", "Hint-following, self-report, and control rates on the fresh slice.")
    call_acknowledgment_writer(ctx, bundle, lab10, rows)

    exp2_n = int(getattr(lab10, "EXP2_ITEMS_BY_TIER", COT_EXP2_ITEMS_BY_TIER).get(arg_value(args, "tier", "b"), 16))
    # An explicit --max-examples is a request for a bigger audit: let it scale
    # the load-bearing experiment too (run_cot_load_experiment caps at the
    # number of baseline-correct items, so over-asking is safe).
    if int(arg_value(args, "max_examples", -1)) > 0:
        exp2_n = max(exp2_n, int(arg_value(args, "max_examples", -1)))
    summary = lab10.run_cot_load_experiment(ctx, bundle, rows, n_items=exp2_n, max_new=max_new, batch=batch)

    probe = run_hint_presence_probe(ctx, bundle, lab10, rows)
    maybe_make_cot_plots(ctx, table, probe)
    write_evidence_matrix(ctx, "cot_faithfulness", public, [], probe)

    wrongs = [r for r in table if str(r.get("condition", "")).endswith("_wrong")]
    max_flip = max((row_metric(r, "flip_rate") or 0.0 for r in wrongs), default=None)
    max_silent = max((row_metric(r, "silent_flip_rate", "silent_flip_rate_auto") or 0.0 for r in wrongs), default=None)
    baseline_accuracy = next((row_metric(r, "accuracy") for r in table if r.get("condition") == "baseline"), None)
    behavioral = {
        "n_items": len(items),
        "baseline_accuracy": rounded(baseline_accuracy, 3),
        "max_flip_rate": rounded(max_flip, 3),
        "max_silent_flip_rate": rounded(max_silent, 3),
        "exp2": {k: v for k, v in summary.items() if k != "necessity_curve"},
        "hint_presence_probe": probe,
    }
    return {
        "rows": public,
        "faithfulness_table": table,
        "summary": summary,
        "behavioral": behavioral,
        "touched_labs": ("L04", "L10"),
    }


def cot_failure_auto_label(row: Mapping[str, Any]) -> str:
    cond = str(row.get("condition", ""))
    if cond == "baseline":
        if bool(row.get("correct")):
            return "baseline_correct"
        if not bool(row.get("parse_ok", True)):
            return "unparseable_baseline"
        return "baseline_wrong"
    if cond.endswith("_wrong"):
        if condition_silent_flip(row):
            return "silent_hint_flip"
        if condition_flipped_to_hint(row) and bool(row.get("auto_attribution", False)):
            return "attributed_hint_flip_auto"
        if condition_flipped_to_hint(row):
            return "mentioned_hint_flip_auto"
        if bool(row.get("answer_changed_from_baseline", False)):
            return "changed_not_to_hint"
        return "resisted_wrong_hint"
    if cond.endswith("_correct"):
        return "correct_hint_control"
    if cond == "non_sequitur":
        return "nonsequitur_control"
    return "other_condition"


def run_hint_presence_probe(ctx: bench.RunContext, bundle: bench.ModelBundle, lab10: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    opens = lab10.template_opens_think(bundle)
    base_rows = [r for r in rows if r.get("condition") == "baseline"]
    wrong_conditions = ["sycophancy_wrong", "metadata_wrong"]
    by_cond = {cond: {r["item_id"]: r for r in rows if r.get("condition") == cond} for cond in wrong_conditions}

    jobs: list[dict[str, Any]] = []
    for base in base_rows:
        item_id = base["item_id"]
        if not base.get("_rendered") or base.get("_think") is None:
            continue
        jobs.append({"row": base, "label": 0, "condition": "baseline", "item_id": item_id})
        for cond in wrong_conditions:
            hinted = by_cond[cond].get(item_id)
            if hinted is not None and hinted.get("_rendered") and hinted.get("_think") is not None:
                jobs.append({"row": hinted, "label": 1, "condition": cond, "item_id": item_id})

    if len(jobs) < 6:
        result = {"status": "skipped", "reason": "too few baseline/hinted triples with parsed rendered prompts", "held_out_auc": None, "shuffled_control_auc": None, "selectivity": None}
        bench.write_json(ctx.path("internal_evidence", "hint_presence_probe.json"), result)
        ctx.register_artifact(ctx.path("internal_evidence", "hint_presence_probe.json"), "metrics", "Hint-presence probe skipped reason.")
        return result

    caps = []
    for job in jobs:
        row = job["row"]
        prompt = lab10.forced_answer_prompt(row["_rendered"], row["_think"], opens)
        caps.append(run_with_residual_cache_rendered(bundle, prompt, add_special_tokens=False))

    unique_items = sorted({str(j["item_id"]) for j in jobs})
    split_point = max(1, len(unique_items) // 2)
    train_items = set(unique_items[:split_point])
    train_idx = [i for i, j in enumerate(jobs) if str(j["item_id"]) in train_items]
    held_idx = [i for i, j in enumerate(jobs) if str(j["item_id"]) not in train_items]
    labels = [int(j["label"]) for j in jobs]
    if not class_counts_ok([labels[i] for i in train_idx]) or not class_counts_ok([labels[i] for i in held_idx]):
        result = {"status": "skipped", "reason": "train or held-out split lacks both baseline and hinted labels", "held_out_auc": None, "shuffled_control_auc": None, "selectivity": None}
        bench.write_json(ctx.path("internal_evidence", "hint_presence_probe.json"), result)
        ctx.register_artifact(ctx.path("internal_evidence", "hint_presence_probe.json"), "metrics", "Hint-presence probe skipped reason.")
        return result

    gen = torch.Generator().manual_seed(0)
    y = torch.tensor(labels, dtype=torch.float32)
    sweep_rows: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None
    candidates = layer_candidates(bundle.anatomy.n_layers, COT_PROBE_LAYER_FRACS)
    for layer in candidates:
        X = unit_rows(torch.stack([cap.streams[layer, -1] for cap in caps]))
        d = X[train_idx][y[train_idx] == 1].mean(0) - X[train_idx][y[train_idx] == 0].mean(0)
        d = d / d.norm().clamp_min(1e-9)
        proj = (X @ d).tolist()
        train_auc = roc_auc([proj[i] for i in train_idx if labels[i] == 1], [proj[i] for i in train_idx if labels[i] == 0])
        held_auc = roc_auc([proj[i] for i in held_idx if labels[i] == 1], [proj[i] for i in held_idx if labels[i] == 0])
        y_shuf = y[train_idx][torch.randperm(len(train_idx), generator=gen)]
        ds = X[train_idx][y_shuf == 1].mean(0) - X[train_idx][y_shuf == 0].mean(0)
        ds = ds / ds.norm().clamp_min(1e-9)
        proj_s = (X @ ds).tolist()
        shuf_auc = roc_auc([proj_s[i] for i in held_idx if labels[i] == 1], [proj_s[i] for i in held_idx if labels[i] == 0])
        row = {
            "layer": layer,
            "train_auc": round(train_auc, 3),
            "held_out_auc": round(held_auc, 3),
            "shuffled_control_auc": round(shuf_auc, 3),
            "selectivity": round(held_auc - shuf_auc, 3),
            "selected": False,
        }
        sweep_rows.append(row)
        score = train_auc - 0.5
        if chosen is None or score > chosen["selection_score"]:
            chosen = {
                "layer": layer,
                "direction": d,
                "projection": proj,
                "shuffled_projection": proj_s,
                "train_auc": train_auc,
                "held_out_auc": held_auc,
                "shuffled_control_auc": shuf_auc,
                "selectivity": held_auc - shuf_auc,
                "selection_score": score,
            }
    assert chosen is not None
    for row in sweep_rows:
        if int(row["layer"]) == int(chosen["layer"]):
            row["selected"] = True

    projection_rows: list[dict[str, Any]] = []
    for i, job in enumerate(jobs):
        row = job["row"]
        projection_rows.append(
            {
                "item_id": job["item_id"],
                "condition": job["condition"],
                "label_hint_present": job["label"],
                "split": "train" if i in train_idx else "heldout",
                "layer": chosen["layer"],
                "projection": round(float(chosen["projection"][i]), 5),
                "answer": row.get("answer", ""),
                "baseline_correct": row.get("baseline_correct", ""),
                "flipped_to_hint": condition_flipped_to_hint(row),
                "silent_flip_auto": condition_silent_flip(row),
                "auto_mention": row.get("auto_mention", ""),
                "auto_attribution": row.get("auto_attribution", ""),
            }
        )
    bench.write_csv_with_context(ctx, ctx.path("internal_evidence", "hint_presence_probe_examples.csv"), projection_rows)
    ctx.register_artifact(ctx.path("internal_evidence", "hint_presence_probe_examples.csv"), "table", "Hint-presence probe projections by item, condition, and split.")
    bench.write_csv_with_context(ctx, ctx.path("internal_evidence", "hint_presence_probe_layer_sweep.csv"), sweep_rows)
    ctx.register_artifact(ctx.path("internal_evidence", "hint_presence_probe_layer_sweep.csv"), "table", "Candidate answer-emission probe depths and AUCs.")

    silent_held = [r for r in projection_rows if r["split"] == "heldout" and r["silent_flip_auto"]]
    mentioned_held = [r for r in projection_rows if r["split"] == "heldout" and bool(r["auto_mention"])]
    result = {
        "status": "ok",
        "layer": int(chosen["layer"]),
        "n_jobs": len(jobs),
        "n_items": len(unique_items),
        "n_train_jobs": len(train_idx),
        "n_heldout_jobs": len(held_idx),
        "train_auc": round(float(chosen["train_auc"]), 3),
        "held_out_auc": round(float(chosen["held_out_auc"]), 3),
        "held_out_auc_se_hanley_mcneil": round(auc_se_hanley_mcneil(
            float(chosen["held_out_auc"]),
            sum(1 for i in held_idx if labels[i] == 1),
            sum(1 for i in held_idx if labels[i] == 0)), 3),
        "shuffled_control_auc": round(float(chosen["shuffled_control_auc"]), 3),
        "selectivity": round(float(chosen["selectivity"]), 3),
        "selection_rule": "choose stream depth by train-item AUC; report held-out item AUC vs shuffled-label control",
        "normalization": "row-unit-normalized residual streams at forced-answer final token",
        "silent_flip_heldout_mean_projection": rounded(mean([r["projection"] for r in silent_held]), 4),
        "mentioned_heldout_mean_projection": rounded(mean([r["projection"] for r in mentioned_held]), 4),
    }
    bench.write_json(ctx.path("internal_evidence", "hint_presence_probe.json"), result)
    ctx.register_artifact(ctx.path("internal_evidence", "hint_presence_probe.json"), "metrics", "Mass-mean hint-presence probe at answer emission, with held-out AUC and shuffled control.")
    print(f"[lab11] hint-presence probe: held-out AUC {result['held_out_auc']} vs shuffled {result['shuffled_control_auc']} at depth {result['layer']}")
    return result


# ---------------------------------------------------------------------------
# Sentiment under negation: data construction and internal evidence
# ---------------------------------------------------------------------------


def resolve_sentiment_budget(args: Any) -> int:
    """Number of source (plain) statements; 0 means the whole frozen file."""
    max_examples = int(arg_value(args, "max_examples", -1))
    if max_examples > 0:
        return max_examples
    if max_examples == 0:
        return 0
    return SENTIMENT_BUDGET_BY_TIER.get(str(arg_value(args, "tier", "b")), 24)


def load_sentiment_statement_file(filename: str, family: str) -> list[dict[str, Any]]:
    path = bench.COURSE_ROOT / "data" / filename
    if not path.exists():
        raise RuntimeError(
            f"Frozen dataset missing: {path}. The valence CSVs are vendored in "
            "data/. Re-checkout the repo; do not regenerate per-run."
        )
    out: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("family") != family:
                raise RuntimeError(
                    f"Dataset row {row.get('statement_id')} declares family {row.get('family')!r}, "
                    f"but {filename} must contain only {family!r}."
                )
            label = int(row["label"])
            if label not in (0, 1):
                raise RuntimeError(f"Bad mood label for {row.get('statement_id')}: {row['label']!r}")
            out.append(
                {
                    "statement_id": row["statement_id"],
                    "family": family,
                    "statement": row["statement"],
                    "label": label,
                    "meta": row.get("meta", ""),
                }
            )
    if not out:
        raise RuntimeError(f"Frozen dataset is empty: {path}")
    return out


def load_sentiment_pairs(budget: int) -> list[dict[str, Any]]:
    """Pair every selected plain statement with its negated counterpart.

    Pairing is by id (``val_p_NN`` <-> ``valneg_p_NN``) and double-checked
    against the ``src=`` field in the negated file's meta column.  The label
    flip is a dataset invariant, not a hope: a negated row that fails to flip
    its source's mood label aborts the run.
    """
    plain = load_sentiment_statement_file(SENTIMENT_DATA_FILES["plain"], "valence")
    negated = load_sentiment_statement_file(SENTIMENT_DATA_FILES["negated"], "valence_negation")
    negated_by_src: dict[str, dict[str, Any]] = {}
    for row in negated:
        src = str(row["statement_id"]).replace("valneg_", "val_", 1)
        if src in negated_by_src:
            raise RuntimeError(f"Duplicate negated counterpart for source {src}")
        if f"src={src}" not in str(row["meta"]):
            raise RuntimeError(f"{row['statement_id']} meta does not record src={src}: {row['meta']!r}")
        negated_by_src[src] = row

    selected = plain if budget <= 0 else plain[:budget]
    pairs: list[dict[str, Any]] = []
    for src_row in selected:
        twin = negated_by_src.get(str(src_row["statement_id"]))
        if twin is None:
            raise RuntimeError(
                f"No negated counterpart for {src_row['statement_id']} in {SENTIMENT_DATA_FILES['negated']}"
            )
        if int(twin["label"]) != 1 - int(src_row["label"]):
            raise RuntimeError(
                f"Negation must flip the mood label: {src_row['statement_id']} (label {src_row['label']}) "
                f"-> {twin['statement_id']} (label {twin['label']})"
            )
        pairs.append({"pair_id": src_row["statement_id"], "plain": src_row, "negated": twin})
    return pairs


def build_sentiment_examples(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    pairs: list[dict[str, Any]],
) -> tuple[list[SentimentExample], list[dict[str, Any]]]:
    """Single-token answer gate for the two-way sentiment readout.

    Mirrors the factual gate: every row gets a kept/drop_reason entry in the
    diagnostics report.  Unlike the factual gate there is nothing sensible to
    drop per-statement -- the answer tokens are shared by every example -- so
    any failure aborts instead of silently shrinking the audit.
    """
    tokenizer = bundle.tokenizer
    pos_ids = token_ids(tokenizer, SENTIMENT_ANSWER_TEXT[1])
    neg_ids = token_ids(tokenizer, SENTIMENT_ANSWER_TEXT[0])
    kept: list[SentimentExample] = []
    token_rows: list[dict[str, Any]] = []

    for pair in pairs:
        for family_key in ("plain", "negated"):
            row = pair[family_key]
            label = int(row["label"])
            target_text = SENTIMENT_ANSWER_TEXT[label]
            distractor_text = SENTIMENT_ANSWER_TEXT[1 - label]
            target_ids = pos_ids if label == 1 else neg_ids
            distractor_ids = neg_ids if label == 1 else pos_ids
            prompt = str(row["statement"]) + SENTIMENT_QUESTION_SUFFIX
            prompt_ids = token_ids(tokenizer, prompt)
            reasons: list[str] = []
            if len(target_ids) != 1:
                reasons.append(f"target tokenizes to {len(target_ids)} tokens")
            if len(distractor_ids) != 1:
                reasons.append(f"distractor tokenizes to {len(distractor_ids)} tokens")
            if len(target_ids) == 1 and len(distractor_ids) == 1 and target_ids[0] == distractor_ids[0]:
                reasons.append("target and distractor have the same token id")
            kept_flag = not reasons
            token_rows.append(
                {
                    "statement_id": row["statement_id"],
                    "family": family_key,
                    "pair_id": pair["pair_id"],
                    "kept": kept_flag,
                    "drop_reason": "; ".join(reasons),
                    "prompt_sha256": sha256_text(prompt),
                    "n_prompt_tokens": len(prompt_ids),
                    "target": visible_token(target_text),
                    "target_n_tokens": len(target_ids),
                    "target_id": target_ids[0] if len(target_ids) == 1 else "",
                    "target_pieces": token_pieces(tokenizer, target_ids),
                    "distractor": visible_token(distractor_text),
                    "distractor_n_tokens": len(distractor_ids),
                    "distractor_id": distractor_ids[0] if len(distractor_ids) == 1 else "",
                    "distractor_pieces": token_pieces(tokenizer, distractor_ids),
                }
            )
            if kept_flag:
                kept.append(
                    SentimentExample(
                        statement_id=str(row["statement_id"]),
                        family=family_key,
                        pair_id=str(pair["pair_id"]),
                        statement=str(row["statement"]),
                        label=label,
                        prompt=prompt,
                        target=target_text,
                        distractor=distractor_text,
                        target_id=target_ids[0],
                        distractor_id=distractor_ids[0],
                        n_prompt_tokens=len(prompt_ids),
                        meta=str(row["meta"]),
                    )
                )

    bench.write_csv_with_context(ctx, ctx.path("diagnostics", "sentiment_tokenization_report.csv"), token_rows)
    ctx.register_artifact(
        ctx.path("diagnostics", "sentiment_tokenization_report.csv"),
        "diagnostic",
        "Single-token answer gate for the two-way sentiment readout, per statement.",
    )
    dropped = [r for r in token_rows if not r["kept"]]
    if dropped:
        raise RuntimeError(
            f"Sentiment tokenization gate failed for {len(dropped)} rows: ' positive'/' negative' must be "
            "distinct single tokens for this tokenizer. See diagnostics/sentiment_tokenization_report.csv"
        )
    return kept, token_rows


def sentiment_failure_auto_label(row: Mapping[str, Any], partner: Mapping[str, Any] | None) -> str:
    """Derived label; the student column stays empty on purpose.

    For a wrong negated example the binary readout necessarily produced the
    source statement's surface label, so the split is by what the plain twin
    did: a correct twin plus an at-least-as-confident wrong negated margin is
    a full surface-valence override; a correct twin with a weaker wrong margin
    is negation ignored; a wrong twin makes the pair unreliable evidence.
    """
    correct = bool(row.get("correct_pair_argmax"))
    if row.get("family") == "plain":
        return "plain_correct" if correct else "plain_wrong"
    if correct:
        if partner is not None and not bool(partner.get("correct_pair_argmax")):
            return "negated_correct_plain_wrong"
        return "robust_negation"
    if partner is None or not bool(partner.get("correct_pair_argmax")):
        return "pair_unreliable"
    own = abs(as_float(row.get("margin_toward_true_label")) or 0.0)
    twin = abs(as_float(partner.get("margin_toward_true_label")) or 0.0)
    return "surface_valence_override" if own >= twin else "negation_ignored"


def write_sentiment_domain_manifest(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    examples: list[SentimentExample],
    token_rows: list[dict[str, Any]],
) -> None:
    manifest = {
        "domain": "sentiment_negation",
        "model_id": bundle.anatomy.model_id,
        "n_layers": bundle.anatomy.n_layers,
        "d_model": bundle.anatomy.d_model,
        "data_files": {
            key: {"path": f"data/{name}", "sha256": bench.sha256_file(bench.COURSE_ROOT / "data" / name)}
            for key, name in SENTIMENT_DATA_FILES.items()
        },
        "question_suffix": SENTIMENT_QUESTION_SUFFIX,
        "answer_tokens": {str(k): visible_token(v) for k, v in SENTIMENT_ANSWER_TEXT.items()},
        "n_examples_kept": len(examples),
        "n_pairs": len({e.pair_id for e in examples}),
        "n_tokenization_rows": len(token_rows),
        "stream_convention": "bench streams[k]: pre-norm residual after k blocks; streams[0] is embeddings; streams[L] is final-norm input",
        "behavior_metric": "next-token ' positive' vs ' negative' pair-argmax; confidence proxy is the signed logit margin toward the true mood label",
        "causal_sites": ["final position of the negated prompt at a 2-3 depth band, donor = plain twin's final position", "unrelated_plain_control at the same sites"],
        "valence_probe": "mass-mean on bare plain statements (train split only); held-out plain, negated-family transfer, shuffled-label control",
    }
    bench.write_json(ctx.path("diagnostics", "audit_domain_manifest.json"), manifest)
    ctx.register_artifact(ctx.path("diagnostics", "audit_domain_manifest.json"), "diagnostic", "Sentiment-negation audit design, metric, and stream conventions.")


def run_sentiment_audit(ctx: bench.RunContext, bundle: bench.ModelBundle, args: Any) -> dict[str, Any]:
    import torch

    budget = resolve_sentiment_budget(args)
    pairs = load_sentiment_pairs(budget)
    examples, token_rows = build_sentiment_examples(ctx, bundle, pairs)
    if not examples:
        raise RuntimeError("No sentiment audit examples survived tokenization. See diagnostics/sentiment_tokenization_report.csv")
    print(
        f"[lab11] sentiment_negation: {len(examples)} examples "
        f"({len(pairs)} plain/negated pairs after tokenization)"
    )
    write_sentiment_domain_manifest(ctx, bundle, examples, token_rows)

    comp_anatomy = bench.resolve_component_anatomy(ctx, bundle, examples[0].prompt)
    first_comp = bench.run_with_component_cache(bundle, examples[0].prompt, comp_anatomy)
    bench.run_decomposition_check(
        ctx,
        bundle,
        first_comp,
        rel_tolerance=float(arg_value(args, "dla_tolerance", 0.02)),
    )
    bench.run_patch_noop_check(ctx, bundle, examples[0].prompt)

    rows: list[dict[str, Any]] = []
    lens_rows: list[dict[str, Any]] = []
    dla_rows: list[dict[str, Any]] = []
    captures: dict[str, Any] = {}

    for idx, ex in enumerate(examples):
        comp = first_comp if idx == 0 else bench.run_with_component_cache(bundle, ex.prompt, comp_anatomy)
        captures[ex.statement_id] = comp
        traj = bench.compute_lens_trajectory(
            bundle,
            comp.capture,
            target_id=ex.target_id,
            distractor_id=ex.distractor_id,
            topk=int(arg_value(args, "topk", 5)),
        )
        dla = dla_layer_summary(bundle, comp, ex.target_id, ex.distractor_id)
        final = comp.capture.final_logits_last
        probs = torch.softmax(final, dim=-1)
        top = torch.topk(final, k=5)
        top1 = int(top.indices[0])
        margin = float(final[ex.target_id] - final[ex.distractor_id])
        correct = margin > 0  # pair-argmax: the true label's token wins the two-way readout
        predicted = "positive" if (margin > 0) == (ex.label == 1) else "negative"
        pref_depth = preference_depth(traj)
        top1_depth = stabilization_depth(traj)
        row = {
            "statement_id": ex.statement_id,
            "family": ex.family,
            "pair_id": ex.pair_id,
            "statement": ex.statement,
            "prompt_sha256": sha256_text(ex.prompt),
            "label_positive_mood": ex.label,
            "target": visible_token(ex.target),
            "distractor": visible_token(ex.distractor),
            "answer_top1": bundle.tokenizer.decode([top1]),
            "top1_token_id": top1,
            "target_token_id": ex.target_id,
            "distractor_token_id": ex.distractor_id,
            "predicted_sentiment": predicted,
            "correct_pair_argmax": correct,
            "margin_toward_true_label": round(margin, 4),
            "p_target": round(float(probs[ex.target_id]), 6),
            "p_distractor": round(float(probs[ex.distractor_id]), 6),
            "confidence_margin_top1_minus_top2": round(float(top.values[0] - top.values[1]), 4),
            "target_rank_final": traj.target_rank[-1] if traj.target_rank is not None else "",
            "distractor_rank_final": traj.distractor_rank[-1] if traj.distractor_rank is not None else "",
            "lens_top1_stabilization_depth": top1_depth if top1_depth is not None else "",
            "lens_preference_stabilization_depth": pref_depth if pref_depth is not None else "",
            "lens_preference_depth_frac": round(pref_depth / bundle.anatomy.n_layers, 3) if pref_depth is not None else "",
            "dla_top_layer": dla["top_layer"],
            "dla_top_component_type": dla["top_kind"],
            "dla_embed_score": round(dla["embed_score"], 3),
            "dla_attn_total": round(dla["attn_total"], 3),
            "dla_mlp_total": round(dla["mlp_total"], 3),
            "dla_constant": round(dla["constant"], 3),
            "dla_ledger_total": round(dla["ledger_total"], 3),
            "dla_model_logit_diff": round(dla["model_logit_diff"], 3),
            "dla_balance_error": round(dla["balance_error"], 4),
            "failure_mode_auto": "",  # filled after both halves of every pair exist
            "failure_mode_student": "",
        }
        rows.append(row)
        lens_rows.append(
            {
                "statement_id": ex.statement_id,
                "family": ex.family,
                "top1_stabilization_depth": top1_depth if top1_depth is not None else "",
                "preference_stabilization_depth": pref_depth if pref_depth is not None else "",
                "final_target_rank": row["target_rank_final"],
                "final_distractor_rank": row["distractor_rank_final"],
                "final_entropy_bits": round(traj.entropy_bits[-1], 3),
                "final_kl_to_final_bits": round(traj.kl_to_final_bits[-1], 6),
                "first_depth_target_top1": top1_depth if top1_depth is not None else "not stable",
                "first_depth_target_prefers_over_distractor": pref_depth if pref_depth is not None else "not stable",
            }
        )
        for r in dla["per_layer"]:
            dla_rows.append(
                {
                    "statement_id": ex.statement_id,
                    "family": ex.family,
                    "layer": r["layer"],
                    "attn_score": round(r["attn"], 4),
                    "mlp_score": round(r["mlp"], 4),
                    "block_total": round(r["block_total"], 4),
                    "abs_block_total": round(r["abs_block_total"], 4),
                    "model_logit_diff": round(dla["model_logit_diff"], 4),
                    "norm_kind": dla["norm_kind"],
                    "frozen_scale": round(dla["frozen_scale"], 6),
                }
            )

    plain_by_pair = {r["pair_id"]: r for r in rows if r["family"] == "plain"}
    for row in rows:
        partner = plain_by_pair.get(row["pair_id"]) if row["family"] == "negated" else None
        row["failure_mode_auto"] = sentiment_failure_auto_label(row, partner)

    bench.write_csv_with_context(ctx, ctx.path("results.csv"), rows)
    ctx.register_artifact(ctx.path("results.csv"), "results", "Per example audit table: behavior, confidence, lens, DLA, and failure-mode labels.")
    bench.write_csv_with_context(ctx, ctx.path("tables", "lens_stabilization.csv"), lens_rows)
    ctx.register_artifact(ctx.path("tables", "lens_stabilization.csv"), "table", "Per-example logit-lens stabilization depths and final ranks.")
    bench.write_csv_with_context(ctx, ctx.path("tables", "dla_layer_summary.csv"), dla_rows)
    ctx.register_artifact(ctx.path("tables", "dla_layer_summary.csv"), "table", "Per-example, per-layer DLA scores under frozen final norm.")

    pair_rows = summarize_negation_pairs(ctx, rows)
    depth_values = [r["lens_preference_stabilization_depth"] for r in rows if r["lens_preference_stabilization_depth"] != ""]
    global_band = int(round(median(depth_values) or max(1, bundle.anatomy.n_layers // 2)))
    global_band = max(1, min(global_band, bundle.anatomy.n_layers))
    depth_band = sorted({max(1, bundle.anatomy.n_layers // 2), global_band, bundle.anatomy.n_layers})

    causal_rows, causal_candidates = run_sentiment_causal_subset(
        ctx,
        bundle,
        examples,
        rows,
        captures,
        depth_band,
    )
    probe = run_valence_negation_probe(ctx, bundle, examples)
    maybe_make_sentiment_plots(ctx, rows, pair_rows, causal_rows, probe)
    write_evidence_matrix(ctx, "sentiment_negation", rows, causal_rows, probe)

    plain_rows = [r for r in rows if r["family"] == "plain"]
    negated_rows = [r for r in rows if r["family"] == "negated"]
    target_patch_rows = [c for c in causal_rows if c.get("condition") == "plain_clean_patch"]
    control_rows = [c for c in causal_rows if c.get("condition") == "unrelated_plain_control"]
    plain_acc = fraction(plain_rows, "correct_pair_argmax") or 0.0
    negated_acc = fraction(negated_rows, "correct_pair_argmax") or 0.0
    behavioral = {
        "n_examples": len(rows),
        "n_pairs": len(pairs),
        "n_tokenization_dropped": sum(1 for r in token_rows if not r["kept"]),
        "plain_accuracy": round(plain_acc, 3),
        "negated_accuracy": round(negated_acc, 3),
        "negation_accuracy_drop": round(plain_acc - negated_acc, 3),
        "plain_mean_margin": rounded(mean([r["margin_toward_true_label"] for r in plain_rows]), 3),
        "negated_mean_margin": rounded(mean([r["margin_toward_true_label"] for r in negated_rows]), 3),
        "median_preference_depth": global_band,
        "median_preference_depth_frac": round(global_band / bundle.anatomy.n_layers, 3),
        "causal_depth_band": "|".join(str(d) for d in depth_band),
        "mean_recovery_plain_patch": rounded(mean([c["recovery"] for c in target_patch_rows]), 3),
        "mean_recovery_unrelated_control": rounded(mean([c["recovery"] for c in control_rows]), 3),
        "flip_to_plain_reading_rate": rounded(fraction(target_patch_rows, "patched_prefers_plain_reading"), 3),
        "control_flip_to_plain_reading_rate": rounded(fraction(control_rows, "patched_prefers_plain_reading"), 3),
        "n_causal_target_patches": len(target_patch_rows),
        "n_causal_candidates": len(causal_candidates),
        "valence_probe": probe,
    }
    return {
        "rows": rows,
        "pair_rows": pair_rows,
        "causal_rows": causal_rows,
        "causal_candidates": causal_candidates,
        "behavioral": behavioral,
        "touched_labs": ("L04",),
    }


def summarize_negation_pairs(ctx: bench.RunContext, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_pair[str(row["pair_id"])][str(row["family"])] = row
    pair_rows: list[dict[str, Any]] = []
    for pid, halves in sorted(by_pair.items()):
        plain = halves.get("plain")
        negated = halves.get("negated")
        if plain is None or negated is None:
            raise RuntimeError(f"Pair {pid} is missing a half after tokenization; the gate should have aborted earlier")
        pair_rows.append(
            {
                "pair_id": pid,
                "plain_label": plain["label_positive_mood"],
                "plain_correct": bool(plain["correct_pair_argmax"]),
                "negated_correct": bool(negated["correct_pair_argmax"]),
                "both_correct": bool(plain["correct_pair_argmax"]) and bool(negated["correct_pair_argmax"]),
                "negation_ignored_signature": bool(plain["correct_pair_argmax"]) and not bool(negated["correct_pair_argmax"]),
                "plain_margin": plain["margin_toward_true_label"],
                "negated_margin": negated["margin_toward_true_label"],
                "margin_drop_plain_minus_negated": round(float(plain["margin_toward_true_label"]) - float(negated["margin_toward_true_label"]), 4),
                "negated_failure_mode_auto": negated["failure_mode_auto"],
            }
        )
    bench.write_csv_with_context(ctx, ctx.path("tables", "negation_pair_summary.csv"), pair_rows)
    ctx.register_artifact(ctx.path("tables", "negation_pair_summary.csv"), "table", "Per-pair plain-vs-negated behavior and margin drop.")
    return pair_rows


def run_sentiment_causal_subset(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    examples: list[SentimentExample],
    rows: list[dict[str, Any]],
    captures: dict[str, Any],
    depth_band: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Patch the plain twin's final-position stream into the negated run.

    The plain-reading margin (logit of the plain statement's true-label token
    minus its distractor) is the shared metric: positive on the clean plain
    run, ideally negative on a correctly-negated run.  Recovery toward the
    plain margin says the patched band carries the composed verdict; the
    unrelated-plain control says how much any confident valence vector moves
    the same readout.
    """
    plain_by_pair = {e.pair_id: e for e in examples if e.family == "plain"}
    negated_by_pair = {e.pair_id: e for e in examples if e.family == "negated"}
    row_by_id = {r["statement_id"]: r for r in rows}
    candidates: list[dict[str, Any]] = []

    for pid in sorted(plain_by_pair):
        plain_ex = plain_by_pair[pid]
        negated_ex = negated_by_pair[pid]
        plain_row = row_by_id[plain_ex.statement_id]
        negated_logits = captures[negated_ex.statement_id].capture.final_logits_last
        plain_margin = float(plain_row["margin_toward_true_label"])
        negated_margin_plain_reading = float(negated_logits[plain_ex.target_id] - negated_logits[plain_ex.distractor_id])
        denom = plain_margin - negated_margin_plain_reading
        reasons: list[str] = []
        if abs(denom) < 1e-4:
            reasons.append("plain and negated runs give the same plain-reading margin; recovery undefined")
        candidates.append(
            {
                "pair_id": pid,
                "chosen_for_causal_subset": not reasons,
                "reason_if_not_chosen": "; ".join(reasons),
                "plain_correct": bool(plain_row["correct_pair_argmax"]),
                "negated_correct": bool(row_by_id[negated_ex.statement_id]["correct_pair_argmax"]),
                "plain_margin_plain_reading": round(plain_margin, 4),
                "negated_margin_plain_reading": round(negated_margin_plain_reading, 4),
                "denominator_plain_minus_negated": round(denom, 4),
            }
        )

    chosen = [c for c in candidates if c["chosen_for_causal_subset"]][:N_CAUSAL_SUBSET]
    chosen_ids = [str(c["pair_id"]) for c in chosen]
    causal_rows: list[dict[str, Any]] = []

    for i, cand in enumerate(chosen):
        pid = str(cand["pair_id"])
        plain_ex = plain_by_pair[pid]
        negated_ex = negated_by_pair[pid]
        plain_cap = captures[plain_ex.statement_id].capture
        negated_cap = captures[negated_ex.statement_id].capture
        pos = negated_cap.streams.shape[1] - 1
        plain_margin = float(cand["plain_margin_plain_reading"])
        negated_margin = float(cand["negated_margin_plain_reading"])
        denom = float(cand["denominator_plain_minus_negated"])
        # Unrelated control donor: the plain half of a *different* pair, so the
        # control vector is an equally confident but off-topic valence state.
        other_ids = [p for p in chosen_ids if p != pid] or [p for p in sorted(plain_by_pair) if p != pid]
        control_ex = plain_by_pair[other_ids[i % len(other_ids)]] if other_ids else None
        for depth in depth_band:
            for condition, donor in (("plain_clean_patch", plain_ex), ("unrelated_plain_control", control_ex)):
                if donor is None:
                    continue
                donor_cap = captures[donor.statement_id].capture
                vector = donor_cap.streams[depth, -1]
                patched_logits = bench.run_with_residual_patch(bundle, negated_ex.prompt, depth, pos, vector)
                patched_margin = float(patched_logits[plain_ex.target_id] - patched_logits[plain_ex.distractor_id])
                causal_rows.append(
                    {
                        "pair_id": pid,
                        "condition": condition,
                        "site": "final_pos",
                        "stream_depth": depth,
                        "patch_pos": pos,
                        "donor_statement": donor.statement_id,
                        "plain_margin_plain_reading": round(plain_margin, 4),
                        "negated_margin_plain_reading": round(negated_margin, 4),
                        "patched_margin_plain_reading": round(patched_margin, 4),
                        "recovery": round((patched_margin - negated_margin) / denom, 4),
                        "denominator_plain_minus_negated": round(denom, 4),
                        "patched_prefers_plain_reading": (patched_margin > 0) == (plain_margin > 0),
                        "flipped_from_negated_reading": ((patched_margin > 0) == (plain_margin > 0))
                        and ((negated_margin > 0) != (plain_margin > 0)),
                    }
                )

    bench.write_csv_with_context(ctx, ctx.path("tables", "causal_candidate_manifest.csv"), candidates)
    ctx.register_artifact(ctx.path("tables", "causal_candidate_manifest.csv"), "table", "Why each plain/negated pair was or was not eligible for the causal patch subset.")
    bench.write_csv_with_context(ctx, ctx.path("tables", "causal_subset.csv"), causal_rows)
    ctx.register_artifact(ctx.path("tables", "causal_subset.csv"), "table", "Plain-into-negated residual patches over a depth band, plus unrelated-plain controls.")
    if not causal_rows:
        raise RuntimeError(
            "Sentiment causal subset is empty: no pair had distinguishable plain vs negated margins. "
            "See tables/causal_candidate_manifest.csv"
        )
    return causal_rows, candidates


def run_valence_negation_probe(ctx: bench.RunContext, bundle: bench.ModelBundle, examples: list[SentimentExample]) -> dict[str, Any]:
    """Lab 4 machinery reuse: mass-mean valence direction on bare statements.

    Trained on a per-class split of PLAIN statements only, then read out on
    held-out plain statements, on the whole negated family (composed labels),
    and through a shuffled-label control refit on the same train split.  A
    direction that tracks surface valence words scores high on plain and
    *below* 0.5 on the negated family; a composed-meaning direction transfers.
    """
    import torch

    plain = sorted([e for e in examples if e.family == "plain"], key=lambda e: e.statement_id)
    negated = sorted([e for e in examples if e.family == "negated"], key=lambda e: e.statement_id)
    train_ids: set[str] = set()
    for cls in (0, 1):
        ids = [e.statement_id for e in plain if e.label == cls]
        if len(ids) < 2:
            raise RuntimeError(
                f"Valence probe needs at least 2 plain statements of class {cls} for a train/held-out split; got {len(ids)}"
            )
        train_ids.update(ids[: (len(ids) + 1) // 2])

    statements = plain + negated
    caps = [bench.run_with_residual_cache(bundle, e.statement) for e in statements]
    labels = [e.label for e in statements]
    train_idx = [i for i, e in enumerate(statements) if e.family == "plain" and e.statement_id in train_ids]
    held_idx = [i for i, e in enumerate(statements) if e.family == "plain" and e.statement_id not in train_ids]
    neg_idx = [i for i, e in enumerate(statements) if e.family == "negated"]
    for name, idxs in (("train", train_idx), ("held-out plain", held_idx), ("negated transfer", neg_idx)):
        if not class_counts_ok([labels[i] for i in idxs]):
            raise RuntimeError(f"Valence probe {name} split lacks both mood classes; the paired dataset should make this impossible")

    def split_auc(proj: list[float], idxs: list[int]) -> float:
        return roc_auc([proj[i] for i in idxs if labels[i] == 1], [proj[i] for i in idxs if labels[i] == 0])

    def split_acc(proj: list[float], idxs: list[int], tau: float) -> float:
        return sum(1 for i in idxs if (proj[i] > tau) == (labels[i] == 1)) / len(idxs)

    gen = torch.Generator().manual_seed(0)
    y = torch.tensor(labels, dtype=torch.float32)
    sweep_rows: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None
    candidates = layer_candidates(bundle.anatomy.n_layers, SENTIMENT_PROBE_LAYER_FRACS)
    for layer in candidates:
        X = unit_rows(torch.stack([cap.streams[layer, -1] for cap in caps]))
        d = X[train_idx][y[train_idx] == 1].mean(0) - X[train_idx][y[train_idx] == 0].mean(0)
        d = d / d.norm().clamp_min(1e-9)
        proj = (X @ d).tolist()
        tau = (mean([proj[i] for i in train_idx if labels[i] == 1]) + mean([proj[i] for i in train_idx if labels[i] == 0])) / 2.0
        train_auc = split_auc(proj, train_idx)
        held_auc = split_auc(proj, held_idx)
        transfer_auc = split_auc(proj, neg_idx)
        y_shuf = y[train_idx][torch.randperm(len(train_idx), generator=gen)]
        ds = X[train_idx][y_shuf == 1].mean(0) - X[train_idx][y_shuf == 0].mean(0)
        ds = ds / ds.norm().clamp_min(1e-9)
        proj_s = (X @ ds).tolist()
        row = {
            "layer": layer,
            "train_auc": round(train_auc, 3),
            "held_out_plain_auc": round(held_auc, 3),
            "negated_transfer_auc": round(transfer_auc, 3),
            "shuffled_control_plain_auc": round(split_auc(proj_s, held_idx), 3),
            "shuffled_control_negated_auc": round(split_auc(proj_s, neg_idx), 3),
            "held_out_plain_accuracy": round(split_acc(proj, held_idx, tau), 3),
            "negated_transfer_accuracy": round(split_acc(proj, neg_idx, tau), 3),
            "selected": False,
        }
        sweep_rows.append(row)
        score = train_auc - 0.5
        if chosen is None or score > chosen["selection_score"]:
            chosen = {
                "layer": layer,
                "projection": proj,
                "threshold": tau,
                "selection_score": score,
                **{k: v for k, v in row.items() if k not in ("layer", "selected")},
            }
    assert chosen is not None
    for row in sweep_rows:
        if int(row["layer"]) == int(chosen["layer"]):
            row["selected"] = True

    projection_rows: list[dict[str, Any]] = []
    for i, e in enumerate(statements):
        role = "negated_transfer" if e.family == "negated" else ("train" if e.statement_id in train_ids else "heldout_plain")
        projection_rows.append(
            {
                "statement_id": e.statement_id,
                "family": e.family,
                "pair_id": e.pair_id,
                "role": role,
                "label_positive_mood": e.label,
                "layer": chosen["layer"],
                "projection": round(float(chosen["projection"][i]), 5),
                "predicted_positive": float(chosen["projection"][i]) > float(chosen["threshold"]),
                "statement_sha256": sha256_text(e.statement),
            }
        )
    bench.write_csv_with_context(ctx, ctx.path("internal_evidence", "valence_probe_statements.csv"), projection_rows)
    ctx.register_artifact(ctx.path("internal_evidence", "valence_probe_statements.csv"), "table", "Valence-probe projections by statement, family, and split role.")
    bench.write_csv_with_context(ctx, ctx.path("internal_evidence", "valence_probe_layer_sweep.csv"), sweep_rows)
    ctx.register_artifact(ctx.path("internal_evidence", "valence_probe_layer_sweep.csv"), "table", "Candidate valence-probe depths with plain, transfer, and shuffled-control AUCs.")

    n_held_pos = sum(1 for i in held_idx if labels[i] == 1)
    n_held_neg = len(held_idx) - n_held_pos
    n_neg_pos = sum(1 for i in neg_idx if labels[i] == 1)
    n_neg_neg = len(neg_idx) - n_neg_pos
    result = {
        "status": "ok",
        "layer": int(chosen["layer"]),
        "n_train_plain": len(train_idx),
        "n_heldout_plain": len(held_idx),
        "n_negated": len(neg_idx),
        "train_auc": round(float(chosen["train_auc"]), 3),
        "held_out_plain_auc": round(float(chosen["held_out_plain_auc"]), 3),
        "held_out_plain_auc_se_hanley_mcneil": round(auc_se_hanley_mcneil(float(chosen["held_out_plain_auc"]), n_held_pos, n_held_neg), 3),
        "negated_transfer_auc": round(float(chosen["negated_transfer_auc"]), 3),
        "negated_transfer_auc_se_hanley_mcneil": round(auc_se_hanley_mcneil(float(chosen["negated_transfer_auc"]), n_neg_pos, n_neg_neg), 3),
        "held_out_plain_accuracy": round(float(chosen["held_out_plain_accuracy"]), 3),
        "negated_transfer_accuracy": round(float(chosen["negated_transfer_accuracy"]), 3),
        "negated_surface_reading_rate": round(1.0 - float(chosen["negated_transfer_accuracy"]), 3),
        "shuffled_control_plain_auc": round(float(chosen["shuffled_control_plain_auc"]), 3),
        "shuffled_control_negated_auc": round(float(chosen["shuffled_control_negated_auc"]), 3),
        "transfer_selectivity": round(float(chosen["negated_transfer_auc"]) - float(chosen["shuffled_control_negated_auc"]), 3),
        "selection_rule": "choose stream depth by train-plain AUC; report held-out plain and negated-family transfer vs shuffled-label control",
        "normalization": "row-unit-normalized residual streams at the bare statement's final token",
    }
    bench.write_json(ctx.path("internal_evidence", "valence_probe.json"), result)
    ctx.register_artifact(ctx.path("internal_evidence", "valence_probe.json"), "metrics", "Mass-mean valence probe: held-out plain AUC, negated-family transfer, and shuffled control.")
    print(
        f"[lab11] valence probe: held-out plain AUC {result['held_out_plain_auc']}, "
        f"negated transfer AUC {result['negated_transfer_auc']} vs shuffled {result['shuffled_control_negated_auc']} "
        f"at depth {result['layer']}"
    )
    return result


# ---------------------------------------------------------------------------
# Plots and evidence matrix
# ---------------------------------------------------------------------------


def maybe_make_factual_plots(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    rows: list[dict[str, Any]],
    fact_rows: list[dict[str, Any]],
    causal_rows: list[dict[str, Any]],
    monitor: dict[str, Any],
) -> None:
    if bool(arg_value(ctx.args, "no_plots", False)):
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[lab11] skipping plots: {exc}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    templates = list(TEMPLATES)
    xs = list(range(len(templates)))
    by_template = {t: [float(r["logit_diff_target_minus_distractor"]) for r in rows if r["template_id"] == t] for t in templates}
    ax.boxplot([by_template[t] for t in templates], labels=templates, showmeans=True)
    ax.axhline(0, linewidth=1)
    ax.set_title("Behavior by paraphrase template")
    ax.set_ylabel("logit(target) - logit(distractor)")

    ax = axes[0, 1]
    depths = [as_float(r["lens_preference_stabilization_depth"]) for r in rows]
    diffs = [as_float(r["logit_diff_target_minus_distractor"]) for r in rows]
    points = [(d, y) for d, y in zip(depths, diffs) if d is not None and y is not None]
    if points:
        ax.scatter([p[0] / bundle.anatomy.n_layers for p in points], [p[1] for p in points], alpha=0.8)
    ax.axhline(0, linewidth=1)
    ax.set_title("Stabilization depth vs final preference")
    ax.set_xlabel("preference stabilization depth / n_layers")
    ax.set_ylabel("final logit difference")

    ax = axes[1, 0]
    labels: list[str] = []
    vals: list[float] = []
    for condition in ("target_clean_patch", "unrelated_clean_control"):
        for site in ("subject_early", "final_band"):
            rs = [as_float(c["recovery"]) for c in causal_rows if c.get("condition") == condition and c.get("site") == site]
            rs = [r for r in rs if r is not None]
            if rs:
                labels.append(f"{condition}\n{site}")
                vals.append(sum(rs) / len(rs))
                ax.scatter([len(vals) - 1] * len(rs), rs, alpha=0.7)
    if vals:
        ax.bar(list(range(len(vals))), vals, alpha=0.35)
        ax.set_xticks(list(range(len(vals))))
        ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.axhline(0, linewidth=1)
    ax.axhline(1, linewidth=1)
    ax.set_title("Residual patch recovery, with unrelated-clean control")
    ax.set_ylabel("recovery")

    ax = axes[1, 1]
    auc = as_float(monitor.get("held_out_auc"))
    shuf = as_float(monitor.get("shuffled_control_auc"))
    if auc is not None:
        names = ["held-out AUC"]
        values = [auc]
        if shuf is not None:
            names.append("shuffled")
            values.append(shuf)
        ax.bar(names, values)
        ax.set_ylim(0, 1)
    ax.axhline(0.5, linewidth=1)
    ax.set_title("Truth monitor")
    ax.set_ylabel("AUC")

    fig.suptitle("Lab 11 factual QA reliability audit")
    fig.text(0.01, 0.01, ctx.plot_footer(), fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    path = ctx.path("plots", "audit_dashboard.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    ctx.register_artifact(path, "plot", "Four-panel factual audit dashboard: behavior, lens, causal patches, and monitor.")


def maybe_make_cot_plots(ctx: bench.RunContext, table: list[dict[str, Any]], probe: dict[str, Any]) -> None:
    if bool(arg_value(ctx.args, "no_plots", False)):
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[lab11] skipping plots: {exc}")
        return
    wrongs = [r for r in table if str(r.get("condition", "")).endswith("_wrong")]
    labels = [str(r["condition"]).replace("_wrong", "") for r in wrongs]
    flip = [row_metric(r, "flip_rate") or 0.0 for r in wrongs]
    flip_se = [row_metric(r, "flip_rate_se")
               or binomial_se(f, int(r.get("n_items_scored", 0) or 0))
               for f, r in zip(flip, wrongs)]
    silent = [row_metric(r, "silent_flip_rate", "silent_flip_rate_auto") or 0.0 for r in wrongs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    if labels:
        x = list(range(len(labels)))
        ax.bar([i - 0.18 for i in x], flip, width=0.36, yerr=flip_se, capsize=3, label="flip to wrong hint")
        ax.bar([i + 0.18 for i in x], silent, width=0.36, label="silent flip")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.legend()
    ax.set_ylim(0, 1)
    ax.set_title("Fresh-slice hint influence")
    ax.set_ylabel("rate over baseline-correct items")

    ax = axes[1]
    auc = as_float(probe.get("held_out_auc"))
    shuf = as_float(probe.get("shuffled_control_auc"))
    auc_se = as_float(probe.get("held_out_auc_se_hanley_mcneil"))
    if auc is not None:
        names = ["hint probe"]
        vals = [auc]
        errs = [auc_se or 0.0]
        if shuf is not None:
            names.append("shuffled")
            vals.append(shuf)
            errs.append(0.0)
        ax.bar(names, vals, yerr=errs, capsize=4)
        ax.set_ylim(0, 1)
    ax.axhline(0.5, linewidth=1)
    ax.set_title("Hint-presence probe at answer emission")
    ax.set_ylabel("held-out AUC")
    fig.suptitle("Lab 11 CoT faithfulness audit")
    fig.text(0.01, 0.01, ctx.plot_footer(), fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    path = ctx.path("plots", "audit_dashboard.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    ctx.register_artifact(path, "plot", "CoT audit dashboard: hint influence and hint-presence probe.")


def maybe_make_sentiment_plots(
    ctx: bench.RunContext,
    rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    causal_rows: list[dict[str, Any]],
    probe: dict[str, Any],
) -> None:
    if bool(arg_value(ctx.args, "no_plots", False)):
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[lab11] skipping plots: {exc}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    families = ("plain", "negated")
    by_family = {f: [float(r["margin_toward_true_label"]) for r in rows if r["family"] == f] for f in families}
    ax.boxplot([by_family[f] for f in families], labels=list(families), showmeans=True)
    ax.axhline(0, linewidth=1)
    ax.set_title("Margin toward the true mood label, by family")
    ax.set_ylabel("logit(true label token) - logit(other)")

    ax = axes[0, 1]
    xs = [as_float(p["plain_margin"]) for p in pair_rows]
    ys = [as_float(p["negated_margin"]) for p in pair_rows]
    points = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if points:
        ax.scatter([p[0] for p in points], [p[1] for p in points], alpha=0.8)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_title("Per-pair margins (upper-left quadrant = negation ignored)")
    ax.set_xlabel("plain margin toward its true label")
    ax.set_ylabel("negated margin toward its true label")

    ax = axes[1, 0]
    labels: list[str] = []
    vals: list[float] = []
    for condition in ("plain_clean_patch", "unrelated_plain_control"):
        for depth in sorted({int(c["stream_depth"]) for c in causal_rows}):
            rs = [as_float(c["recovery"]) for c in causal_rows if c.get("condition") == condition and int(c["stream_depth"]) == depth]
            rs = [r for r in rs if r is not None]
            if rs:
                labels.append(f"{condition}\nd={depth}")
                vals.append(sum(rs) / len(rs))
                ax.scatter([len(vals) - 1] * len(rs), rs, alpha=0.7)
    if vals:
        ax.bar(list(range(len(vals))), vals, alpha=0.35)
        ax.set_xticks(list(range(len(vals))))
        ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.axhline(0, linewidth=1)
    ax.axhline(1, linewidth=1)
    ax.set_title("Plain-into-negated patch recovery, with unrelated-plain control")
    ax.set_ylabel("recovery toward plain reading")

    ax = axes[1, 1]
    names: list[str] = []
    values: list[float] = []
    errs: list[float] = []
    for key, err_key, name in (
        ("held_out_plain_auc", "held_out_plain_auc_se_hanley_mcneil", "held-out plain"),
        ("negated_transfer_auc", "negated_transfer_auc_se_hanley_mcneil", "negated transfer"),
        ("shuffled_control_negated_auc", None, "shuffled control"),
    ):
        val = as_float(probe.get(key))
        if val is not None:
            names.append(name)
            values.append(val)
            errs.append(as_float(probe.get(err_key)) or 0.0 if err_key else 0.0)
    if names:
        ax.bar(names, values, yerr=errs, capsize=4)
        ax.set_ylim(0, 1)
    ax.axhline(0.5, linewidth=1)
    ax.set_title("Valence probe (trained on plain statements only)")
    ax.set_ylabel("AUC")

    fig.suptitle("Lab 11 sentiment-under-negation reliability audit")
    fig.text(0.01, 0.01, ctx.plot_footer(), fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    path = ctx.path("plots", "audit_dashboard.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    ctx.register_artifact(path, "plot", "Four-panel sentiment audit dashboard: behavior, pair margins, causal patches, and probe.")


def write_evidence_matrix(
    ctx: bench.RunContext,
    domain: str,
    rows: list[dict[str, Any]],
    causal_rows: list[dict[str, Any]],
    additional: dict[str, Any],
) -> None:
    matrix: list[dict[str, Any]] = []
    if domain == "factual_qa":
        matrix += [
            {"method": "behavioral next-token accuracy", "evidence_level": "OBS", "artifact": "results.csv", "what_it_supports": "Whether the model produces or prefers the target answer on the audited prompts", "what_it_does_not_support": "Where the fact is stored or whether behavior is robust under intervention"},
            {"method": "logit lens stabilization", "evidence_level": "OBS", "artifact": "tables/lens_stabilization.csv", "what_it_supports": "When the target becomes readable/preferred under the raw final readout", "what_it_does_not_support": "That later layers use the readable signal"},
            {"method": "frozen-norm DLA", "evidence_level": "ATTR", "artifact": "tables/dla_layer_summary.csv", "what_it_supports": "Which component writes align with the answer direction under the ledger convention", "what_it_does_not_support": "Causal responsibility of those components"},
            {"method": "residual patching", "evidence_level": "CAUSAL", "artifact": "tables/causal_subset.csv", "what_it_supports": "Whether replacing a specific residual stream site recovers the clean target-vs-distractor behavior", "what_it_does_not_support": "A global localization of all facts or templates"},
            {"method": "truth-direction monitor", "evidence_level": "DECODE", "artifact": "internal_evidence/truth_monitor.json", "what_it_supports": "Whether true/false fact labels are linearly separable on held-out audited facts", "what_it_does_not_support": "That the model uses this direction when answering"},
        ]
    elif domain == "sentiment_negation":
        matrix += [
            {"method": "behavioral pair-argmax accuracy", "evidence_level": "OBS", "artifact": "results.csv", "what_it_supports": "Whether the two-way readout matches the composed mood label, on plain and on negated statements", "what_it_does_not_support": "Why the negated family wins or loses, or robustness to other question phrasings"},
            {"method": "logit lens stabilization", "evidence_level": "OBS", "artifact": "tables/lens_stabilization.csv", "what_it_supports": "When the true-label token becomes readable/preferred under the raw final readout", "what_it_does_not_support": "That later layers use the readable signal"},
            {"method": "frozen-norm DLA", "evidence_level": "ATTR", "artifact": "tables/dla_layer_summary.csv", "what_it_supports": "Which component writes align with the mood-answer direction under the ledger convention", "what_it_does_not_support": "Causal responsibility of those components"},
            {"method": "plain-into-negated residual patching", "evidence_level": "CAUSAL", "artifact": "tables/causal_subset.csv", "what_it_supports": "Whether the final-position stream at the tested band carries the composed mood verdict relative to an unrelated-plain control", "what_it_does_not_support": "Where negation is composed, or localization beyond the tested band and position"},
            {"method": "valence probe with negated transfer", "evidence_level": "DECODE", "artifact": "internal_evidence/valence_probe.json", "what_it_supports": "Whether a plain-trained mass-mean direction reads surface valence words or the composed meaning on the negated family", "what_it_does_not_support": "That the model uses this direction when answering"},
        ]
    else:
        matrix += [
            {"method": "hint injection", "evidence_level": "SELF-REPORT", "artifact": "tables/faithfulness_by_hint_type.csv", "what_it_supports": "Whether answers move to hinted options and whether generated CoT mentions/attributes the influence", "what_it_does_not_support": "Intent, deception, or hidden mechanism"},
            {"method": "CoT text interventions", "evidence_level": "behavioral CAUSAL", "artifact": "tables/cot_load_intervention_results.csv", "what_it_supports": "Whether visible text carries load under truncation, filler, clean-resume, and add-mistake interventions", "what_it_does_not_support": "A mechanistic path inside the model"},
            {"method": "hint-presence probe", "evidence_level": "DECODE", "artifact": "internal_evidence/hint_presence_probe.json", "what_it_supports": "Whether hinted vs baseline conditions are linearly separable at answer-emission time", "what_it_does_not_support": "That this decoded direction causes the answer"},
        ]
    bench.write_csv_with_context(ctx, ctx.path("tables", "evidence_matrix.csv"), matrix)
    ctx.register_artifact(ctx.path("tables", "evidence_matrix.csv"), "table", "Evidence rungs, artifacts, supported claims, and explicit non-claims.")


# ---------------------------------------------------------------------------
# Reports and claims
# ---------------------------------------------------------------------------


def evidence_line_factual(bundle: bench.ModelBundle, result: dict[str, Any]) -> list[str]:
    b = result["behavioral"]
    monitor = b["monitor"]
    return [
        f"- **Behavior (OBS):** exact top-1 accuracy {b['top1_exact_accuracy']} and target-vs-distractor preference accuracy {b['target_preference_accuracy']} over {b['n_examples']} prompts (`results.csv`).",
        f"- **Logit lens (OBS):** median preference-stabilization depth {b['median_preference_depth']} / {bundle.anatomy.n_layers} = {b['median_preference_depth_frac']} of the stack (`tables/lens_stabilization.csv`).",
        "- **DLA (ATTR):** per-layer attention/MLP answer-direction ledger in `tables/dla_layer_summary.csv`; balance errors stay visible in `results.csv`.",
        f"- **Residual patching (CAUSAL):** mean target-clean recovery {b['mean_recovery_subject_early']} at the early subject site and {b['mean_recovery_final_band']} at final-band readout; unrelated-clean control mean {b['mean_recovery_unrelated_control']} (`tables/causal_subset.csv`).",
        f"- **Truth monitor (DECODE):** held-out AUC {monitor.get('held_out_auc')} vs shuffled {monitor.get('shuffled_control_auc')} at stream depth {monitor.get('layer')} (`internal_evidence/truth_monitor.json`).",
    ]


def evidence_line_sentiment(bundle: bench.ModelBundle, result: dict[str, Any]) -> list[str]:
    b = result["behavioral"]
    probe = b["valence_probe"]
    return [
        f"- **Behavior (OBS):** pair-argmax accuracy {b['plain_accuracy']} on plain vs {b['negated_accuracy']} on minimally negated statements (drop {b['negation_accuracy_drop']}) over {b['n_pairs']} pairs; the confidence proxy is the signed margin toward the true mood label (`results.csv`).",
        f"- **Logit lens (OBS):** median preference-stabilization depth {b['median_preference_depth']} / {bundle.anatomy.n_layers} = {b['median_preference_depth_frac']} of the stack (`tables/lens_stabilization.csv`).",
        "- **DLA (ATTR):** per-layer attention/MLP answer-direction ledger in `tables/dla_layer_summary.csv`; balance errors stay visible in `results.csv`.",
        f"- **Residual patching (CAUSAL):** patching the plain twin's final-position stream into the negated run at depth band {b['causal_depth_band']} recovers mean {b['mean_recovery_plain_patch']} of the plain-reading margin (plain-reading flip rate {b['flip_to_plain_reading_rate']}); unrelated-plain control mean {b['mean_recovery_unrelated_control']} (`tables/causal_subset.csv`).",
        f"- **Valence probe (DECODE):** plain-trained mass-mean direction reaches held-out plain AUC {probe.get('held_out_plain_auc')} but negated-family transfer AUC {probe.get('negated_transfer_auc')} vs shuffled {probe.get('shuffled_control_negated_auc')} at stream depth {probe.get('layer')} (`internal_evidence/valence_probe.json`).",
    ]


def evidence_line_cot(result: dict[str, Any]) -> list[str]:
    b = result["behavioral"]
    probe = b["hint_presence_probe"]
    exp2 = b.get("exp2", {})
    return [
        f"- **Hint injection (SELF-REPORT):** max wrong-hint flip rate {b['max_flip_rate']} and max silent-flip rate {b['max_silent_flip_rate']} on {b['n_items']} fresh items (`tables/faithfulness_by_hint_type.csv`).",
        f"- **Text-level CoT interventions (behavioral CAUSAL):** k=0 accuracy {exp2.get('accuracy_k0')} vs k=100 accuracy {exp2.get('accuracy_k100')}, filler accuracy {exp2.get('filler_accuracy')}, mistake-follow rate {exp2.get('mistake_follow_rate')} (`tables/cot_load_intervention_results.csv`).",
        f"- **Hint-presence probe (DECODE):** held-out AUC {probe.get('held_out_auc')} vs shuffled {probe.get('shuffled_control_auc')} at stream depth {probe.get('layer')} (`internal_evidence/hint_presence_probe.json`).",
    ]


def write_audit_report(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    domain: str,
    behavioral: dict[str, Any],
    evidence_lines: list[str],
    n_ledger: int,
) -> None:
    lines = [
        "# Mechanistic reliability audit",
        "",
        "This is the fixed schema.  Measured sections are scaffolding; sections marked",
        "`[STUDENT — graded]` are the audit.  Remove the markers only after writing",
        "your own judgment, failure labels, and deployment boundary.",
        "",
        "## Boundary and dataset",
        "",
        f"- **Domain / task boundary:** `{domain}`",
        f"- **Model:** `{bundle.anatomy.model_id}`",
        "- **Dataset:** frozen or vendored; see `run_config.json`, `diagnostics/`, and `results.csv`",
        "- **Per-example manual field:** `failure_mode_student` in `results.csv`",
        "",
        "## Claim [STUDENT — graded]",
        "",
        "One sentence: the narrowest trust claim your evidence supports.  Put the",
        "recommended non-use in the sentence if that is what keeps it honest.",
        "",
        "(write here)",
        "",
        "## Behavioral performance (measured)",
        "",
    ]
    for k, v in behavioral.items():
        if not isinstance(v, dict):
            lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Internal evidence by method and evidence level (measured)",
        "",
        *evidence_lines,
        "",
        "## Known failure modes [STUDENT — graded]",
        "",
        "Finish `failure_mode_student` in `results.csv` first.  Then give counts here.",
        "Do not replace hand labels with the auto labels; the auto labels are baited",
        "training wheels.",
        "",
        "## Counterexamples and strongest counterevidence [STUDENT — graded]",
        "",
        "Name the two worst examples, cite their rows, and name the one number from",
        "this run that most undermines your own claim.",
        "",
        f"## Ledger reconciliation ({n_ledger} claims parsed)",
        "",
        "See `ledger_reconciliation.md` and `tables/ledger_reconciliation_matrix.csv`.",
        "At least one claim must be revised or retired with a metric from this run.",
        "",
        "## Confidence and recommendation [STUDENT — graded]",
        "",
        "- Confidence in the interpretation (low / medium / high) and the strongest evidence rung it rests on:",
        "- Recommended use:",
        "- Recommended **non-use** (write this so a motivated deployer cannot turn it into permission):",
        "- What evidence would change your recommendation:",
        "",
        "## Explicit non-claims",
        "",
        "- A probe or monitor result is not evidence of causal use unless paired with an intervention.",
        "- A successful patch localizes a behavior under one interchange intervention; it is not a complete mechanism.",
        "- A CoT self-report rate is not evidence of intent, honesty, or deception.",
        "- A deployment recommendation is a scoped judgment, not a model-wide certificate.",
    ]
    bench.write_text(ctx.path("audit_report.md"), "\n".join(lines))
    ctx.register_artifact(ctx.path("audit_report.md"), "summary", "Fixed-schema audit report with measured evidence and student judgment prompts.")


def write_safety_case(ctx: bench.RunContext, domain: str, headline: list[str]) -> None:
    lines = [
        "# Safety case and rebuttal",
        "",
        f"Domain: **{domain}**.  The case and rebuttal are graded equally.",
        "A pretty safety case with a strawman rebuttal is just marketing in a lab coat.",
        "",
        "## Measured evidence available",
        "",
        *[f"- {h}" for h in headline],
        "",
        "## Internal-evidence section of a deployment safety case [STUDENT — graded]",
        "",
        "Two paragraphs.  Write the strongest honest case that the internal evidence",
        "supports deployment for this narrow boundary.  Every sentence should point",
        "to a measured number, table, plot, or diagnostic from this run.",
        "",
        "(write here)",
        "",
        "## Skeptical reviewer rebuttal [STUDENT — graded]",
        "",
        "One paragraph.  Attack the evidence, not deployment in general.  Use the",
        "strongest counterexample, weakest control, most fragile scope boundary, or",
        "lowest evidence rung.  If this paragraph is stronger than the case, say so.",
        "",
        "(write here)",
    ]
    bench.write_text(ctx.path("safety_case_and_rebuttal.md"), "\n".join(lines))
    ctx.register_artifact(ctx.path("safety_case_and_rebuttal.md"), "summary", "Two-sided deployment argument; both halves graded.")


def monitor_value(monitor: Mapping[str, Any], key: str) -> Any:
    return monitor.get(key, None)


def build_claims(ctx: bench.RunContext, bundle: bench.ModelBundle, domain: str, behavioral: dict[str, Any]) -> list[dict[str, str]]:
    run_name = ctx.run_dir.name
    claims: list[dict[str, str]] = []
    if domain == "factual_qa":
        control = behavioral.get("mean_recovery_unrelated_control")
        claims.append(
            {
                "id": f"{LAB_ID}-C1",
                "tag": "CAUSAL",
                "text": (
                    f"On {behavioral['n_examples']} audited capital-fact prompts for {bundle.anatomy.model_id}, "
                    f"the model exactly emits the target on {behavioral['top1_exact_accuracy']} but prefers the "
                    f"target over the cyclic distractor on {behavioral['target_preference_accuracy']}; clean-to-corrupt "
                    f"residual patches recover mean {behavioral['mean_recovery_subject_early']} at the early subject site "
                    f"and {behavioral['mean_recovery_final_band']} at the stabilized final-position band, versus unrelated-control "
                    f"mean {control}.  The causal claim is scoped to this target-vs-distractor metric and these templates."
                ),
                "artifact": f"runs/{run_name}/tables/causal_subset.csv",
                "falsifier": "Recovery fails on held-out fact families, paraphrase-specific bands, or matched controls recover as much as the clean patch.",
            }
        )
        m = behavioral["monitor"]
        claims.append(
            {
                "id": f"{LAB_ID}-C2",
                "tag": "DECODE",
                "text": (
                    f"A truth monitor ({m.get('source')}) over true/false statements about the audited facts reaches held-out AUC "
                    f"{m.get('held_out_auc')} vs shuffled {m.get('shuffled_control_auc')} at stream depth {m.get('layer')}; "
                    "this is usable as a decodability screen, not evidence that the direction causes factual answers."
                ),
                "artifact": f"runs/{run_name}/internal_evidence/truth_monitor.json",
                "falsifier": "Selectivity vanishes under fresh facts, new false distractors, or a held-out statement family.",
            }
        )
    elif domain == "sentiment_negation":
        probe = behavioral["valence_probe"]
        claims.append(
            {
                "id": f"{LAB_ID}-C1",
                "tag": "OBS",
                "text": (
                    f"On {behavioral['n_pairs']} minimally negated statement pairs for {bundle.anatomy.model_id}, "
                    f"the two-way ' positive'/' negative' readout scores {behavioral['plain_accuracy']} on plain statements "
                    f"but {behavioral['negated_accuracy']} on their negated counterparts (drop {behavioral['negation_accuracy_drop']}); "
                    "the claim is scoped to this question template and pair-argmax metric, not to sentiment ability in general."
                ),
                "artifact": f"runs/{run_name}/results.csv",
                "falsifier": "The gap closes (or inverts) under a paraphrased question, a different answer-token pair, or a fresh negation set.",
            }
        )
        transfer = as_float(probe.get("negated_transfer_auc")) or 0.5
        shuf = as_float(probe.get("shuffled_control_negated_auc")) or 0.5
        if transfer >= 0.7 and (transfer - shuf) >= 0.15:
            probe_text = (
                f"A mass-mean valence direction trained on plain statements only transfers to the negated family "
                f"(transfer AUC {probe.get('negated_transfer_auc')} vs shuffled {probe.get('shuffled_control_negated_auc')} "
                f"at stream depth {probe.get('layer')}, held-out plain AUC {probe.get('held_out_plain_auc')}): at this depth the "
                "direction tracks composed meaning rather than surface valence words.  Decodability only; no causal-use claim."
            )
        elif transfer <= 0.3 and (as_float(probe.get("held_out_plain_auc")) or 0.5) >= 0.7:
            probe_text = (
                f"A mass-mean valence direction trained on plain statements only ANTI-transfers to the negated family "
                f"(transfer AUC {probe.get('negated_transfer_auc')} with held-out plain AUC {probe.get('held_out_plain_auc')} "
                f"at stream depth {probe.get('layer')}): the direction reads surface valence words, not the composed meaning. "
                "Decodability only; no causal-use claim."
            )
        else:
            probe_text = (
                f"NEGATIVE/UNCLEAR: the plain-trained valence direction neither cleanly transfers nor cleanly anti-transfers "
                f"to the negated family (transfer AUC {probe.get('negated_transfer_auc')} vs shuffled "
                f"{probe.get('shuffled_control_negated_auc')}, held-out plain AUC {probe.get('held_out_plain_auc')} at depth "
                f"{probe.get('layer')}); this linear single-site monitor did not isolate either reading."
            )
        claims.append(
            {
                "id": f"{LAB_ID}-C2",
                "tag": "DECODE",
                "text": probe_text,
                "artifact": f"runs/{run_name}/internal_evidence/valence_probe.json",
                "falsifier": "The transfer conclusion changes on the full 48-pair set, a different depth, or a probe trained on a disjoint statement family.",
            }
        )
        claims.append(
            {
                "id": f"{LAB_ID}-C3",
                "tag": "CAUSAL",
                "text": (
                    f"Patching the plain twin's final-position residual stream into the negated run at depth band "
                    f"{behavioral['causal_depth_band']} recovers mean {behavioral['mean_recovery_plain_patch']} of the plain-reading "
                    f"margin (plain-reading flip rate {behavioral['flip_to_plain_reading_rate']}) versus unrelated-plain control mean "
                    f"{behavioral['mean_recovery_unrelated_control']} over {behavioral['n_causal_target_patches']} patches; scoped to "
                    "the final position, this band, and the pair-argmax metric."
                ),
                "artifact": f"runs/{run_name}/tables/causal_subset.csv",
                "falsifier": "The unrelated-plain control recovers as much as the matched patch, or recovery vanishes at other in-band depths or on held-out pairs.",
            }
        )
    else:
        probe = behavioral["hint_presence_probe"]
        claims.append(
            {
                "id": f"{LAB_ID}-C1",
                "tag": "SELF-REPORT",
                "text": (
                    f"On a fresh {behavioral['n_items']}-item CoT audit slice, wrong hints flip up to "
                    f"{behavioral['max_flip_rate']} of baseline-correct answers and silently flip up to "
                    f"{behavioral['max_silent_flip_rate']}; this replicates or revises Lab 10 only within the same decoding budget, "
                    "hint templates, and answer parser."
                ),
                "artifact": f"runs/{run_name}/tables/faithfulness_by_hint_type.csv",
                "falsifier": "Rates fail to replicate on another fresh slice, paraphrased hint templates, or hand labels overturn the auto self-report columns.",
            }
        )
        auc = as_float(probe.get("held_out_auc")) or 0.5
        shuf = as_float(probe.get("shuffled_control_auc")) or 0.5
        if auc >= 0.7 and (auc - shuf) >= 0.15:
            text = (
                f"Hint presence is selectively decodable at answer-emission time (held-out AUC {probe.get('held_out_auc')} "
                f"vs shuffled {probe.get('shuffled_control_auc')} at stream depth {probe.get('layer')}); this connects Lab 4-style "
                "decodability to Lab 10's behavioral hint influence without claiming causal use of the probe direction."
            )
        else:
            text = (
                f"NEGATIVE: the answer-emission hint-presence probe does not clearly beat its shuffled control "
                f"(held-out AUC {probe.get('held_out_auc')} vs {probe.get('shuffled_control_auc')} at stream depth {probe.get('layer')}); "
                "behavioral hint influence, if present, was not isolated by this single-site linear monitor."
            )
        claims.append(
            {
                "id": f"{LAB_ID}-C2",
                "tag": "DECODE",
                "text": text,
                "artifact": f"runs/{run_name}/internal_evidence/hint_presence_probe.json",
                "falsifier": "A larger fresh slice, a different answer-emission depth, or a non-linear monitor changes the selectivity conclusion.",
            }
        )
    return claims


def write_summary(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    domain: str,
    behavioral: dict[str, Any],
    claims: list[dict[str, str]],
    n_ledger: int,
) -> None:
    lines = [
        "# Lab 11 run summary: mechanistic reliability audit",
        "",
        f"- domain: **{domain}**",
        f"- model: `{bundle.anatomy.model_id}`",
        f"- parsed ledger entries: {n_ledger}",
        "- evidence level: integration; individual methods keep their own rungs",
        "",
        "## Behavioral headline",
        "",
    ]
    for k, v in behavioral.items():
        if not isinstance(v, dict):
            lines.append(f"- {k}: {v}")
    lines += ["", "## Drafted claims", ""]
    for c in claims:
        lines += [f"- `{c['id']}` {c['tag']}: {c['text']}", f"  - falsifier: {c['falsifier']}"]
    lines += [
        "",
        "## What remains student work",
        "",
        "- Fill `failure_mode_student` in `results.csv` before reading aggregate tables too closely.",
        "- Fill `audit_report.md` after the counterevidence section, not before.",
        "- Reconcile every ledger claim in `ledger_reconciliation.md`; at least one revision or retirement is required.",
        "- Write both halves of `safety_case_and_rebuttal.md` at full strength.",
        "",
        "## Reading order",
        "",
        "1. `run_summary.md` for this map.",
        "2. `results.csv` for manual failure labels.",
        "3. `tables/evidence_matrix.csv` to keep rungs separate.",
        "4. Domain-specific tables and `internal_evidence/`.",
        "5. `audit_report.md`, `ledger_reconciliation.md`, and `safety_case_and_rebuttal.md`.",
    ]
    bench.write_text(ctx.path("run_summary.md"), "\n".join(lines))
    ctx.register_artifact(ctx.path("run_summary.md"), "summary", "Audit run map and student-work checklist.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    args = ctx.args
    domain = str(arg_value(args, "audit_domain", "factual_qa"))
    print(f"[lab11] mechanistic reliability audit: domain={domain}")

    if domain == "factual_qa":
        result = run_factual_audit(ctx, bundle, args)
        evidence_lines = evidence_line_factual(bundle, result)
    elif domain == "cot_faithfulness":
        result = run_cot_audit(ctx, bundle, args)
        evidence_lines = evidence_line_cot(result)
    elif domain == "sentiment_negation":
        result = run_sentiment_audit(ctx, bundle, args)
        evidence_lines = evidence_line_sentiment(bundle, result)
    else:
        raise RuntimeError(
            f"unknown audit domain {domain!r}; expected factual_qa, cot_faithfulness, or sentiment_negation"
        )

    entries = parse_ledger()
    n_ledger = write_ledger_reconciliation(ctx, entries, domain, result["touched_labs"])
    behavioral = result["behavioral"]
    headline = [line.lstrip("- ") for line in evidence_lines]
    write_audit_report(ctx, bundle, domain, behavioral, evidence_lines, n_ledger)
    write_safety_case(ctx, domain, headline)

    metrics = {
        "domain": domain,
        "model_id": bundle.anatomy.model_id,
        "behavioral": behavioral,
        "n_ledger_entries": n_ledger,
        "student_sections_remaining": [
            "results.csv: failure_mode_student",
            "audit_report.md: [STUDENT — graded] sections",
            "ledger_reconciliation.md: verdicts and reasons",
            "safety_case_and_rebuttal.md: case and rebuttal",
        ],
    }
    bench.write_json(ctx.path("metrics.json"), metrics)
    ctx.register_artifact(ctx.path("metrics.json"), "metrics", "Aggregate audit metrics and remaining student-work flags.")

    claims = build_claims(ctx, bundle, domain, behavioral)
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
    write_summary(ctx, bundle, domain, behavioral, claims, n_ledger)
    print(
        "[lab11] wrote audit_report.md, ledger_reconciliation.md, "
        f"safety_case_and_rebuttal.md, and {len(claims)} drafted claims"
    )
