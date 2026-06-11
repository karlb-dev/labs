"""Lab 11: Mechanistic reliability audit (capstone).

Given behavioral evidence and internal evidence, where should we trust this
model less — and what may we responsibly say? The capstone is a plug-in
audit harness with a RIGID output contract (so audits are comparable across
students) and a free choice of domain and depth. It is built on the claim
ledger: the report must cite ledger entries, retire at least one claim that
no longer survives, and add final claims with evidence tags. Retirement is
graded as positively as confirmation.

Two domains are implemented end to end; both follow the same per-example
contract from COURSE.md — model answer + confidence proxy, a logit-lens
stabilization summary, a DLA summary, at least one causal intervention on a
subset, one additional internal method (probe / monitor), and a failure-mode
label (auto-drafted, hand-finished):

* ``--audit-domain factual_qa`` (default) — factual recall under paraphrase,
  continuing Lab 5's dataset on the course base model. Internal methods:
  lens stabilization depth, per-layer DLA of the answer direction, residual
  patching at the stabilization band (causal), and a truth-direction monitor
  over true/false statements about the same facts (Lab 4's saved direction
  when compatible, a mass-mean direction with shuffled control otherwise).

* ``--audit-domain cot_faithfulness`` (recommended flagship) — continues Lab
  10 on a FRESH item slice with a reasoning model (pass
  ``--model allenai/Olmo-3-7B-Think``; Tier A: ``--model Qwen/Qwen3-0.6B``).
  Reuses Lab 10's machinery verbatim: hint injection with controls,
  the thought-necessity curve, add-mistake and filler (behavioral-causal),
  plus a NEW mechanistic method — a "hint present" probe on activations at
  the answer-emission position, with a shuffled-label control: is the hint's
  influence visible internally even when the text never mentions it?

The audit report (audit_report.md) follows the fixed schema; sections that
must be the student's own prose are marked ``[STUDENT — graded]`` and left
as prompts, with the run's measured numbers already cited around them. The
same applies to safety_case_and_rebuttal.md and ledger_reconciliation.md —
the harness assembles the evidence and the scaffolding; the judgment is the
coursework.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import interp_bench as bench

LAB_ID = "L11"

# ---------------------------------------------------------------------------
# Domain A data: facts under paraphrase (continues Lab 5's dataset)
# ---------------------------------------------------------------------------

TEMPLATES = {
    "base": "The capital of {X} is",
    "para_city": "The capital city of {X} is",
    "para_in": "In {X}, the capital is",
}

FACT_POOL: tuple[tuple[str, str, str], ...] = (
    # (id, subject, capital); the corrupt partner for patching/distractors is
    # the NEXT entry (cyclic), which keeps every pair template-aligned.
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
)

N_CAUSAL_SUBSET = 6          # pairs patched in the causal pass
MONITOR_LAYER_FRACS = (0.5, 0.7)


@dataclasses.dataclass
class AuditExample:
    fact_id: str
    template_id: str
    prompt: str
    subject: str
    target: str             # " Paris"
    distractor: str         # the partner fact's capital, " Berlin"
    target_id: int = -1
    distractor_id: int = -1


# ---------------------------------------------------------------------------
# Ledger reconciliation (shared by both domains)
# ---------------------------------------------------------------------------

LEDGER_ENTRY_RE = re.compile(
    r"^\[(?P<id>L\d{2}-C\d+)\]\s+(?P<tag>OBS|ATTR|DECODE|CAUSAL|SELF-REPORT)\s*\|\s*(?P<text>.+)$")


def parse_ledger() -> list[dict[str, str]]:
    if not bench.LEDGER_PATH.exists():
        return []
    entries = []
    lines = bench.LEDGER_PATH.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        m = LEDGER_ENTRY_RE.match(line.strip())
        if not m:
            continue
        artifact = falsifier = ""
        if i + 1 < len(lines) and lines[i + 1].startswith("Artifact:"):
            tail = lines[i + 1][len("Artifact:"):]
            artifact, _, falsifier = tail.partition("| Falsifier:")
        entries.append({"id": m["id"], "tag": m["tag"], "text": m["text"].strip(),
                        "artifact": artifact.strip(), "falsifier": falsifier.strip()})
    return entries


def write_ledger_reconciliation(ctx, entries: list[dict[str, str]], domain: str,
                                touched_labs: tuple[str, ...]) -> int:
    lines = [
        "# Ledger reconciliation",
        "",
        "Every claim in your ledger gets a verdict: **keep** (survives this audit),",
        "**revise** (survives with a narrower scope — write the new scope), or",
        "**retire** (the audit's evidence undercuts it — write what killed it).",
        "Retirement with a sound reason earns the same credit as confirmation.",
        "At least one claim must be retired or revised; a semester of perfect",
        "claims means the falsifier columns were never honest.",
        "",
    ]
    if not entries:
        lines += [
            "**Your claim ledger is empty.** The capstone audits the ledger; without",
            "entries there is nothing to reconcile. Each lab drafted claims into its",
            "run's `ledger_suggestions.md` — go back, edit the ones you would defend,",
            "append them to `claim_ledger.md` (or re-run with `--append-ledger`), then",
            "re-run this audit.",
            "",
        ]
    touched = [e for e in entries if any(e["id"].startswith(lab) for lab in touched_labs)]
    others = [e for e in entries if e not in touched]
    if touched:
        lines += [f"## Claims this audit's evidence touches directly ({domain})", ""]
        for e in touched:
            lines += [f"### {e['id']} ({e['tag']})", "", f"> {e['text']}", "",
                      f"- recorded falsifier: {e['falsifier'] or '(none recorded)'}",
                      "- **verdict [STUDENT — graded]:** keep / revise / retire",
                      "- **reason [STUDENT — graded]:** cite an artifact from THIS run", ""]
    if others:
        lines += ["## Remaining claims (reconcile against the whole semester)", ""]
        for e in others:
            lines += [f"- `{e['id']}` {e['tag']}: {e['text'][:120]}…" if len(e["text"]) > 120
                      else f"- `{e['id']}` {e['tag']}: {e['text']}",
                      "  - verdict [STUDENT]: keep / revise / retire — reason:", ""]
    bench.write_text(ctx.path("ledger_reconciliation.md"), "\n".join(lines))
    ctx.register_artifact(ctx.path("ledger_reconciliation.md"), "summary",
                          "Per-claim keep/revise/retire worksheet (verdicts are graded coursework).")
    return len(entries)


def write_safety_case(ctx, domain: str, headline: list[str]) -> None:
    lines = [
        "# Safety case and rebuttal",
        "",
        f"Domain: **{domain}**. Both halves are graded with equal weight; a rebuttal",
        "written as a strawman scores as if the case had no rebuttal at all.",
        "",
        "## The measured evidence available to you",
        "",
    ]
    lines += [f"- {h}" for h in headline]
    lines += [
        "",
        "## Internal-evidence section of a deployment safety case [STUDENT — graded]",
        "",
        "Two paragraphs. Write the strongest honest case that the internal evidence",
        "above supports deploying this model for this narrow domain. Every sentence",
        "must cite a number or an artifact from this run. Sentences that could have",
        "been written without running the audit score zero.",
        "",
        "(write here)",
        "",
        "## The skeptical reviewer's rebuttal [STUDENT — graded]",
        "",
        "One paragraph. The strongest objection a hostile, competent reviewer would",
        "file — aimed at your evidence, not at deployment in general. If your",
        "rebuttal is stronger than your case, say so in the final line and state",
        "what additional evidence would change the balance.",
        "",
        "(write here)",
        "",
    ]
    bench.write_text(ctx.path("safety_case_and_rebuttal.md"), "\n".join(lines))
    ctx.register_artifact(ctx.path("safety_case_and_rebuttal.md"), "summary",
                          "The two-sided deployment argument (both halves graded).")


# ---------------------------------------------------------------------------
# Shared internal-evidence helpers
# ---------------------------------------------------------------------------


def stabilization_depth(traj: Any) -> int | None:
    """First depth at which the target is top-1 and STAYS top-1 to the end —
    the lens summary the audit reports per example."""
    if traj.target_rank is None:
        return None
    depth = None
    for d in range(traj.n_depths):
        if traj.target_rank[d] == 1:
            if depth is None:
                depth = d
        else:
            depth = None
    return depth


def dla_layer_summary(bundle, comp, target_id: int, distractor_id: int) -> dict[str, Any]:
    """Per-layer contribution of attn/MLP writes to the answer direction,
    linearized with the final norm's observed scale (Lab 2's convention,
    abbreviated): contribution ≈ ((write − mean(write)) · γ/σ_final) · ΔW_U."""
    import torch

    final_stream = comp.capture.streams[-1, -1]
    sigma = torch.sqrt(final_stream.var(unbiased=False) + getattr(bundle.final_norm, "eps", 1e-5))
    gamma = bundle.final_norm.weight.detach().to("cpu", torch.float32)
    w_u = bundle.lm_head.weight.detach()
    direction = (w_u[target_id] - w_u[distractor_id]).to("cpu", torch.float32)
    scaled_dir = gamma * direction / float(sigma)

    def score(vec: Any) -> float:
        return float((vec - vec.mean()) @ scaled_dir)

    per_layer = [{"layer": layer,
                  "attn": round(score(comp.attn_contrib[layer]), 3),
                  "mlp": round(score(comp.mlp_contrib[layer]), 3)}
                 for layer in range(bundle.anatomy.n_layers)]
    best = max(per_layer, key=lambda r: abs(r["attn"]) + abs(r["mlp"]))
    mlp_total = sum(r["mlp"] for r in per_layer)
    attn_total = sum(r["attn"] for r in per_layer)
    return {"per_layer": per_layer, "top_layer": best["layer"],
            "mlp_total": round(mlp_total, 3), "attn_total": round(attn_total, 3)}


def roc_auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return 0.5
    wins = ties = 0
    for p in pos:
        for q in neg:
            wins += p > q
            ties += p == q
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


# ---------------------------------------------------------------------------
# Domain A: factual QA under paraphrase
# ---------------------------------------------------------------------------


def build_factual_examples(bundle, max_facts: int) -> tuple[list[AuditExample], list[dict[str, Any]]]:
    """Facts x templates, single-token validated, with the cyclic partner's
    capital as the distractor. Returns (kept, dropped-with-reasons)."""
    tok = bundle.tokenizer
    pool = FACT_POOL[:max_facts] if max_facts > 0 else FACT_POOL
    kept, dropped = [], []
    for i, (fid, subject, capital) in enumerate(pool):
        partner = pool[(i + 1) % len(pool)]
        target, distractor = " " + capital, " " + partner[2]
        tid = tok(target)["input_ids"]
        did = tok(distractor)["input_ids"]
        if len(tid) != 1 or len(did) != 1:
            dropped.append({"fact_id": fid, "reason": "multi-token answer"})
            continue
        for template_id, template in TEMPLATES.items():
            kept.append(AuditExample(fact_id=fid, template_id=template_id,
                                     prompt=template.format(X=subject), subject=subject,
                                     target=target, distractor=distractor,
                                     target_id=tid[0], distractor_id=did[0]))
    return kept, dropped


def run_factual_audit(ctx, bundle, args) -> dict[str, Any]:
    import torch

    examples, dropped = build_factual_examples(bundle, args.max_examples)
    print(f"[lab11] factual_qa: {len(examples)} examples "
          f"({len(set(e.fact_id for e in examples))} facts x {len(TEMPLATES)} templates)")
    comp_anatomy = bench.resolve_component_anatomy(ctx, bundle, examples[0].prompt)
    bench.run_patch_noop_check(ctx, bundle, examples[0].prompt)

    # ---- behavioral + lens + DLA per example -------------------------------
    rows = []
    captures: dict[tuple[str, str], Any] = {}
    for ex in examples:
        comp = bench.run_with_component_cache(bundle, ex.prompt, comp_anatomy)
        captures[(ex.fact_id, ex.template_id)] = comp
        traj = bench.compute_lens_trajectory(bundle, comp.capture,
                                             target_id=ex.target_id, distractor_id=ex.distractor_id)
        dla = dla_layer_summary(bundle, comp, ex.target_id, ex.distractor_id)
        final = comp.capture.final_logits_last
        probs = torch.softmax(final, dim=-1)
        top1 = int(final.argmax())
        answer = bundle.tokenizer.decode([top1])
        correct = top1 == ex.target_id
        # auto failure-mode draft; the student column stays empty on purpose
        if correct:
            failure_auto = ""
        elif top1 == ex.distractor_id:
            failure_auto = "distractor_win"
        elif answer.strip().istitle() and len(answer.strip()) > 2:
            failure_auto = "other_entity"
        else:
            failure_auto = "non_answer_token"
        rows.append({
            "fact_id": ex.fact_id, "template_id": ex.template_id, "prompt": ex.prompt,
            "target": ex.target, "distractor": ex.distractor,
            "answer_top1": answer, "correct": correct,
            "p_target": round(float(probs[ex.target_id]), 5),
            "logit_diff": round(float(final[ex.target_id] - final[ex.distractor_id]), 4),
            "confidence_margin": round(float(torch.topk(final, 2).values[0]
                                             - torch.topk(final, 2).values[1]), 4),
            "lens_stabilization_depth": stabilization_depth(traj),
            "dla_top_layer": dla["top_layer"],
            "dla_mlp_total": dla["mlp_total"], "dla_attn_total": dla["attn_total"],
            "failure_mode_auto": failure_auto, "failure_mode_student": "",
        })
    bench.write_csv_with_context(ctx, ctx.path("results.csv"), rows)
    ctx.register_artifact(ctx.path("results.csv"), "results",
                          "Per example: behavior, confidence, lens depth, DLA summary, failure draft.")

    # paraphrase consistency per fact
    by_fact: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_fact.setdefault(r["fact_id"], []).append(r)
    fact_rows = [{"fact_id": fid,
                  "n_templates": len(rs),
                  "n_correct": sum(r["correct"] for r in rs),
                  "consistent": len({r["answer_top1"] for r in rs}) == 1,
                  "min_logit_diff": min(r["logit_diff"] for r in rs),
                  "stabilization_spread": (max(r["lens_stabilization_depth"] or 0 for r in rs)
                                           - min(r["lens_stabilization_depth"] or 0 for r in rs))}
                 for fid, rs in by_fact.items()]
    bench.write_csv_with_context(ctx, ctx.path("tables", "paraphrase_consistency.csv"), fact_rows)
    ctx.register_artifact(ctx.path("tables", "paraphrase_consistency.csv"), "table",
                          "Per-fact accuracy and answer consistency across templates.")

    # ---- causal subset: patch the stabilization band ------------------------
    correct_base = [e for e in examples if e.template_id == "base"
                    and next(r for r in rows if r["fact_id"] == e.fact_id
                             and r["template_id"] == "base")["correct"]]
    depths = [r["lens_stabilization_depth"] for r in rows
              if r["lens_stabilization_depth"] is not None and r["correct"]]
    band = sorted(depths)[len(depths) // 2] if depths else bundle.anatomy.n_layers // 2
    band = max(1, min(band, bundle.anatomy.n_layers - 1))
    causal_rows = []
    for ex in correct_base[:N_CAUSAL_SUBSET]:
        partner = next((p for p in correct_base if p.target == ex.distractor), None)
        if partner is None:
            continue
        clean = captures[(ex.fact_id, "base")].capture
        subj_positions = [i for i, t in enumerate(clean.tokens_text) if t == " " + ex.subject]
        corrupt_prompt = partner.prompt
        corrupt_ids = bundle.tokenizer(corrupt_prompt)["input_ids"]
        if len(corrupt_ids) != len(clean.input_ids) or not subj_positions:
            continue
        base_logits = bench.next_token_logits(bundle, corrupt_prompt)
        base_diff = float(base_logits[ex.target_id] - base_logits[ex.distractor_id])
        clean_row = next(r for r in rows if r["fact_id"] == ex.fact_id and r["template_id"] == "base")
        denom = clean_row["logit_diff"] - base_diff
        # Lab 5's localization lesson, applied: subject-token patches matter
        # EARLY (where recall happens), final-token patches at the band
        # (where the readout has stabilized). The audit tests both sites.
        early = max(1, band // 2)
        for site, depth, pos in (("subject_early", early, subj_positions[-1]),
                                 ("final_band", band, len(clean.input_ids) - 1)):
            patched = bench.run_with_residual_patch(bundle, corrupt_prompt, depth, pos,
                                                    clean.streams[depth, pos])
            patched_diff = float(patched[ex.target_id] - patched[ex.distractor_id])
            causal_rows.append({
                "clean_fact": ex.fact_id, "corrupt_fact": partner.fact_id,
                "site": site, "patch_depth": depth, "patch_pos": pos,
                "corrupt_logit_diff": round(base_diff, 3),
                "patched_logit_diff": round(patched_diff, 3),
                "clean_logit_diff": clean_row["logit_diff"],
                "recovery": round((patched_diff - base_diff) / denom, 3) if abs(denom) > 1e-6 else "",
            })
    bench.write_csv_with_context(ctx, ctx.path("tables", "causal_subset.csv"), causal_rows)
    ctx.register_artifact(ctx.path("tables", "causal_subset.csv"), "table",
                          "Residual patches at the lens-stabilization band (the audit's causal rung).")

    # ---- truth-direction monitor --------------------------------------------
    monitor = run_truth_monitor(ctx, bundle, examples, rows)

    behavioral = {
        "n_examples": len(rows), "n_dropped": len(dropped),
        "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 3),
        "paraphrase_consistent_facts": round(sum(f["consistent"] for f in fact_rows) / len(fact_rows), 3),
        "median_stabilization_depth": band,
        **{f"mean_recovery_{site}": round(
            sum(c["recovery"] for c in causal_rows if c["site"] == site and c["recovery"] != "") /
            max(1, len([c for c in causal_rows if c["site"] == site and c["recovery"] != ""])), 3)
            for site in ("subject_early", "final_band")},
        "monitor": monitor,
    }
    return {"rows": rows, "fact_rows": fact_rows, "causal_rows": causal_rows,
            "behavioral": behavioral, "band": band,
            "touched_labs": ("L01", "L02", "L04", "L05")}


def run_truth_monitor(ctx, bundle, examples, rows) -> dict[str, Any]:
    """A truth monitor over statements about the audited facts: project the
    statement-final activation onto a truth direction (Lab 4's saved artifact
    when compatible, else a mass-mean direction trained on half the facts)
    and score true-vs-false separation on the held-out half, against a
    shuffled-label control."""
    import torch

    base_examples = [e for e in examples if e.template_id == "base"]
    statements = []
    for e in base_examples:
        statements.append({"fact_id": e.fact_id, "text": f"The capital of {e.subject} is{e.target}.",
                           "label": 1})
        statements.append({"fact_id": e.fact_id, "text": f"The capital of {e.subject} is{e.distractor}.",
                           "label": 0})
    layer = int(bundle.anatomy.n_layers * MONITOR_LAYER_FRACS[1])
    acts, labels, fids = [], [], []
    for s in statements:
        cap = bench.run_with_residual_cache(bundle, s["text"])
        acts.append(cap.streams[layer, -1])
        labels.append(s["label"])
        fids.append(s["fact_id"])
    X = torch.stack(acts)
    y = torch.tensor(labels, dtype=torch.float32)

    unique = sorted(set(fids))
    train_f = set(unique[: len(unique) // 2])
    tr = [i for i, f in enumerate(fids) if f in train_f]
    te = [i for i, f in enumerate(fids) if f not in train_f]

    source = "mass-mean (trained on half the facts)"
    direction = None
    saved = sorted((bench.COURSE_ROOT / "runs").glob("lab04*/tables/truth_direction.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if saved:
        meta = torch.load(saved[0], map_location="cpu", weights_only=False)
        vec = meta.get("direction")
        if vec is not None and len(vec) == X.shape[1]:
            direction = torch.as_tensor(vec, dtype=torch.float32)
            source = f"Lab 4 artifact ({saved[0].parent.parent.name}, layer {meta.get('layer')})"
    if direction is None:
        direction = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
    direction = direction / direction.norm().clamp_min(1e-9)

    proj = (X @ direction).tolist()
    auc = roc_auc([proj[i] for i in te if y[i] == 1], [proj[i] for i in te if y[i] == 0])
    gen = torch.Generator().manual_seed(0)
    y_shuf = y[tr][torch.randperm(len(tr), generator=gen)]
    d_shuf = X[tr][y_shuf == 1].mean(0) - X[tr][y_shuf == 0].mean(0)
    d_shuf = d_shuf / d_shuf.norm().clamp_min(1e-9)
    proj_s = (X @ d_shuf).tolist()
    auc_shuf = roc_auc([proj_s[i] for i in te if y[i] == 1], [proj_s[i] for i in te if y[i] == 0])

    result = {"source": source, "layer": layer, "n_statements": len(statements),
              "held_out_auc": round(auc, 3), "shuffled_control_auc": round(auc_shuf, 3),
              "selectivity": round(auc - auc_shuf, 3)}
    bench.write_json(ctx.path("internal_evidence", "truth_monitor.json"), result)
    ctx.register_artifact(ctx.path("internal_evidence", "truth_monitor.json"), "metrics",
                          "Truth-direction monitor AUC on held-out facts vs shuffled control.")
    print(f"[lab11]   truth monitor ({source}): held-out AUC {auc:.3f} vs shuffled {auc_shuf:.3f}")
    return result


# ---------------------------------------------------------------------------
# Domain B: CoT faithfulness audit (continues Lab 10 on fresh items)
# ---------------------------------------------------------------------------


def run_cot_audit(ctx, bundle, args) -> dict[str, Any]:
    import torch

    import labs.lab10_cot_faithfulness as lab10

    if not bench.supports_chat_template(bundle):
        raise RuntimeError(
            "The cot_faithfulness domain needs a reasoning model. Pass "
            "--model allenai/Olmo-3-7B-Think (Tier B) or --model Qwen/Qwen3-0.6B (smoke).")
    n_items = args.max_examples if args.max_examples > 0 else 12
    all_items = lab10.load_items(0)
    # FRESH slice: offset by one so the audit never reuses Lab 10's default
    # stride-sampled items — auditing the training set of your own beliefs
    # is the failure mode this course exists to unlearn.
    items = all_items[1::4][:n_items]
    max_new = lab10.MAX_NEW_BY_TIER.get(args.tier, 1024)
    batch = lab10.BATCH_BY_TIER.get(args.tier, 8)
    print(f"[lab11] cot_faithfulness: {len(items)} FRESH items (offset slice), "
          f"max_new {max_new}, batch {batch}")

    lab10.run_think_roundtrip_check(ctx, bundle, items, max_new, batch)
    rows = lab10.run_hint_experiment(ctx, bundle, items, max_new=max_new, batch=batch)
    public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    bench.write_csv_with_context(ctx, ctx.path("results.csv"), public)
    ctx.register_artifact(ctx.path("results.csv"), "results",
                          "Per item x condition on the fresh audit slice.")
    table = lab10.faithfulness_table(rows)
    bench.write_csv_with_context(ctx, ctx.path("tables", "faithfulness_by_hint_type.csv"), table)
    ctx.register_artifact(ctx.path("tables", "faithfulness_by_hint_type.csv"), "table",
                          "Flip / acknowledgment rates on the fresh slice, with controls.")
    lab10.write_acknowledgment_samples(ctx, bundle, rows)
    summary = lab10.run_cot_load_experiment(ctx, bundle, rows,
                                            n_items=lab10.EXP2_ITEMS_BY_TIER.get(args.tier, 8),
                                            max_new=max_new, batch=batch)

    # ---- the mechanistic method: a "hint present" probe ---------------------
    # Activations at the answer-emission position (the forced-answer prompt's
    # final token), hinted vs baseline, family-split by item, vs a shuffled
    # control. Decodable hint-presence at answer time — even on items whose
    # CoT never mentions the hint — is Lab 4's lesson closing the course.
    opens = lab10.template_opens_think(bundle)
    layer = int(bundle.anatomy.n_layers * 0.7)
    feats, labels, item_ids = [], [], []
    base_rows = [r for r in rows if r["condition"] == "baseline"]
    hint_rows = {r["item_id"]: r for r in rows if r["condition"] == "sycophancy_wrong"}
    for r in base_rows:
        h = hint_rows.get(r["item_id"])
        if h is None:
            continue
        for row, label in ((r, 0), (h, 1)):
            prompt = lab10.forced_answer_prompt(row["_rendered"], row["_think"], opens)
            cap = bench.run_with_residual_cache(bundle, prompt, add_special_tokens=False)
            feats.append(cap.streams[layer, -1])
            labels.append(label)
            item_ids.append(r["item_id"])
    X = torch.stack(feats)
    y = torch.tensor(labels, dtype=torch.float32)
    unique = sorted(set(item_ids))
    train_set = set(unique[: len(unique) // 2])
    tr = [i for i, f in enumerate(item_ids) if f in train_set]
    te = [i for i, f in enumerate(item_ids) if f not in train_set]
    d = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
    d = d / d.norm().clamp_min(1e-9)
    proj = (X @ d).tolist()
    auc = roc_auc([proj[i] for i in te if y[i] == 1], [proj[i] for i in te if y[i] == 0])
    gen = torch.Generator().manual_seed(0)
    y_shuf = y[tr][torch.randperm(len(tr), generator=gen)]
    ds = X[tr][y_shuf == 1].mean(0) - X[tr][y_shuf == 0].mean(0)
    ds = ds / ds.norm().clamp_min(1e-9)
    proj_s = (X @ ds).tolist()
    auc_shuf = roc_auc([proj_s[i] for i in te if y[i] == 1], [proj_s[i] for i in te if y[i] == 0])
    probe = {"layer": layer, "n_pairs": len(base_rows), "held_out_auc": round(auc, 3),
             "shuffled_control_auc": round(auc_shuf, 3), "selectivity": round(auc - auc_shuf, 3)}
    bench.write_json(ctx.path("internal_evidence", "hint_presence_probe.json"), probe)
    ctx.register_artifact(ctx.path("internal_evidence", "hint_presence_probe.json"), "metrics",
                          "Mass-mean 'hint present' probe at answer emission, vs shuffled control.")
    print(f"[lab11]   hint-presence probe: held-out AUC {auc:.3f} vs shuffled {auc_shuf:.3f}")

    wrongs = [r for r in table if r["condition"].endswith("_wrong") and r.get("flip_rate") != ""]
    behavioral = {
        "n_items": len(items),
        "baseline_accuracy": next((r["accuracy"] for r in table if r["condition"] == "baseline"), None),
        "max_flip_rate": max((r["flip_rate"] for r in wrongs), default=None),
        "max_silent_flip_rate": max((r["silent_flip_rate"] for r in wrongs
                                     if r["silent_flip_rate"] != ""), default=None),
        "exp2": {k: v for k, v in summary.items() if k != "necessity_curve"},
        "hint_presence_probe": probe,
    }
    return {"rows": public, "table": table, "summary": summary, "behavioral": behavioral,
            "touched_labs": ("L04", "L10")}


# ---------------------------------------------------------------------------
# The audit report (fixed schema, both domains)
# ---------------------------------------------------------------------------


def write_audit_report(ctx, bundle, domain: str, behavioral: dict[str, Any],
                       evidence_lines: list[str], n_ledger: int) -> None:
    lines = [
        "# Mechanistic reliability audit",
        "",
        "The fixed schema. Sections marked [STUDENT — graded] are the coursework;",
        "everything else is assembled from this run's measurements and is cited by",
        "artifact so a reviewer can re-derive it.",
        "",
        f"- **Domain / task boundary:** {domain}",
        f"- **Model:** `{bundle.anatomy.model_id}`",
        f"- **Dataset:** frozen, vendored (see run_config.json and results.csv)",
        "",
        "## Claim [STUDENT — graded]",
        "",
        "One sentence: the narrowest claim about where this model can be trusted",
        "that your evidence actually supports. If it takes two sentences, the",
        "domain was scoped too broadly.",
        "",
        "(write here)",
        "",
        "## Behavioral performance (measured)",
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in behavioral.items() if not isinstance(v, dict)]
    lines += [
        "",
        "## Internal evidence, by method and evidence level (measured)",
        "",
    ]
    lines += evidence_lines
    lines += [
        "",
        "## Known failure modes [STUDENT — graded]",
        "",
        "Finish the `failure_mode_student` column in results.csv first; then name",
        "the modes here with counts. The auto column is a draft, not a finding.",
        "",
        "## Counterexamples and strongest counterevidence [STUDENT — graded]",
        "",
        "The two worst examples in results.csv, by name, and the single measured",
        "number from this run that most undermines your claim.",
        "",
        f"## Ledger reconciliation ({n_ledger} claims parsed)",
        "",
        "See ledger_reconciliation.md — at least one claim must be revised or",
        "retired, with the artifact that killed it.",
        "",
        "## Confidence and recommendation [STUDENT — graded]",
        "",
        "- Confidence in the interpretation (low/medium/high) and the evidence rung",
        "  it rests on:",
        "- Recommended use:",
        "- Recommended NON-use (a boundary a motivated deployer could not misread):",
        "",
    ]
    bench.write_text(ctx.path("audit_report.md"), "\n".join(lines))
    ctx.register_artifact(ctx.path("audit_report.md"), "summary",
                          "The fixed-schema audit report (student sections marked).")


def build_claims(ctx, bundle, domain, behavioral) -> list[dict[str, str]]:
    run_name = ctx.run_dir.name
    claims = []
    if domain == "factual_qa":
        claims.append({
            "id": f"{LAB_ID}-C1", "tag": "CAUSAL",
            "text": (
                f"On {behavioral['n_examples']} capital-fact prompts (accuracy "
                f"{behavioral['accuracy']}, paraphrase-consistent on "
                f"{behavioral['paraphrase_consistent_facts']} of facts), patching clean residuals "
                f"into the corrupt run recovers a mean {behavioral['mean_recovery_subject_early']} "
                f"of the logit gap at the early subject site and "
                f"{behavioral['mean_recovery_final_band']} at the final position of the "
                f"stabilization band (depth {behavioral['median_stabilization_depth']}) — recall "
                f"and readout localized where Lab 5 said they would be, on this dataset."
            ),
            "artifact": f"runs/{run_name}/tables/causal_subset.csv",
            "falsifier": "Recovery collapses at the same depth on held-out facts, or the band shifts across paraphrases.",
        })
        m = behavioral["monitor"]
        claims.append({
            "id": f"{LAB_ID}-C2", "tag": "DECODE",
            "text": (
                f"A truth-direction monitor ({m['source']}) separates true from false statements "
                f"about the audited facts at held-out AUC {m['held_out_auc']} vs shuffled control "
                f"{m['shuffled_control_auc']} (selectivity {m['selectivity']}) at layer {m['layer']} — "
                f"internal evidence usable as a screen, not proof of belief (Lab 4's standards apply)."
            ),
            "artifact": f"runs/{run_name}/internal_evidence/truth_monitor.json",
            "falsifier": "Selectivity vanishes on a fresh statement family or a different paraphrase of the same facts.",
        })
    else:
        p = behavioral["hint_presence_probe"]
        claims.append({
            "id": f"{LAB_ID}-C1", "tag": "SELF-REPORT",
            "text": (
                f"On a FRESH {behavioral['n_items']}-item slice (baseline accuracy "
                f"{behavioral['baseline_accuracy']}), the strongest hint flips up to "
                f"{behavioral['max_flip_rate']} of baseline-correct answers with a silent-flip rate "
                f"of {behavioral['max_silent_flip_rate']} — replicating Lab 10's finding out of "
                f"sample, which is what an audit is."
            ),
            "artifact": f"runs/{run_name}/tables/faithfulness_by_hint_type.csv",
            "falsifier": "Rates fail to replicate on another fresh slice or under paraphrased hint templates.",
        })
        claims.append({
            "id": f"{LAB_ID}-C2", "tag": "DECODE",
            "text": (
                f"Hint presence is decodable from activations at the answer-emission position "
                f"(held-out AUC {p['held_out_auc']} vs shuffled {p['shuffled_control_auc']}, layer "
                f"{p['layer']}) — the influence the CoT may omit in text is visible internally, "
                f"connecting Lab 4's decodability machinery to Lab 10's behavioral finding."
            ),
            "artifact": f"runs/{run_name}/internal_evidence/hint_presence_probe.json",
            "falsifier": "AUC drops to the shuffled control on more items, or the direction fails to transfer across hint types.",
        })
    return claims


def write_summary(ctx, bundle, domain, behavioral, claims, n_ledger) -> None:
    lines = [
        "# Lab 11 run summary: mechanistic reliability audit",
        "",
        f"- domain: **{domain}** | model: `{bundle.anatomy.model_id}`",
        f"- ledger entries parsed for reconciliation: {n_ledger}",
        "",
        "## Behavioral headline",
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in behavioral.items() if not isinstance(v, dict)]
    lines += ["", "## Drafted claims", ""]
    for c in claims:
        lines.append(f"- `{c['id']}` {c['tag']}: {c['text']}")
        lines.append(f"  - falsifier: {c['falsifier']}")
    lines += [
        "",
        "## What the harness did NOT do",
        "",
        "- It did not write your claim, your failure-mode labels, your verdicts, your",
        "  safety case, or your rebuttal. Those files contain [STUDENT — graded]",
        "  markers; an audit with the markers still in place is not an audit.",
        "",
        "## Reading order",
        "",
        "1. `audit_report.md` — fill in the student sections LAST, after 2–5.",
        "2. `results.csv` — label the failure modes by hand.",
        "3. `tables/` and `internal_evidence/` — the measured rungs.",
        "4. `ledger_reconciliation.md` — keep / revise / retire, with reasons.",
        "5. `safety_case_and_rebuttal.md` — both halves, graded equally.",
        "",
    ]
    bench.write_text(ctx.path("run_summary.md"), "\n".join(lines))
    ctx.register_artifact(ctx.path("run_summary.md"), "summary", "The audit run's map.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    args = ctx.args
    domain = args.audit_domain
    print(f"[lab11] mechanistic reliability audit — domain: {domain}")

    if domain == "factual_qa":
        result = run_factual_audit(ctx, bundle, args)
        b = result["behavioral"]
        evidence_lines = [
            f"- **Logit lens (OBS):** median stabilization depth {b['median_stabilization_depth']} "
            f"of {bundle.anatomy.n_layers}; per-example depths in results.csv.",
            f"- **DLA (ATTR):** per-example top layer and attn/MLP split in results.csv.",
            f"- **Residual patching (CAUSAL):** mean recovery {b['mean_recovery_subject_early']} "
            f"(early subject site) / {b['mean_recovery_final_band']} (final position, stabilization "
            f"band) over {len(result['causal_rows'])} patches (tables/causal_subset.csv).",
            f"- **Truth monitor (DECODE):** held-out AUC {b['monitor']['held_out_auc']} vs shuffled "
            f"{b['monitor']['shuffled_control_auc']} (internal_evidence/truth_monitor.json).",
        ]
    elif domain == "cot_faithfulness":
        result = run_cot_audit(ctx, bundle, args)
        b = result["behavioral"]
        evidence_lines = [
            f"- **Hint injection (SELF-REPORT):** max flip {b['max_flip_rate']}, max silent flip "
            f"{b['max_silent_flip_rate']} on the fresh slice (tables/faithfulness_by_hint_type.csv).",
            f"- **Text-level interventions (behavioral CAUSAL):** necessity "
            f"{b['exp2'].get('accuracy_k0')}→{b['exp2'].get('accuracy_k100')}, filler "
            f"{b['exp2'].get('filler_accuracy')}, mistake-follow {b['exp2'].get('mistake_follow_rate')}.",
            f"- **Hint-presence probe (DECODE):** held-out AUC {b['hint_presence_probe']['held_out_auc']} "
            f"vs shuffled {b['hint_presence_probe']['shuffled_control_auc']} at answer emission "
            f"(internal_evidence/hint_presence_probe.json).",
        ]
    else:
        raise RuntimeError(f"unknown audit domain {domain!r}")

    entries = parse_ledger()
    n_ledger = write_ledger_reconciliation(ctx, entries, domain, result["touched_labs"])
    behavioral = result["behavioral"]
    headline = [line.lstrip("- ") for line in evidence_lines]
    write_audit_report(ctx, bundle, domain, behavioral, evidence_lines, n_ledger)
    write_safety_case(ctx, domain, headline)

    metrics = {"domain": domain, "model_id": bundle.anatomy.model_id,
               "behavioral": behavioral, "n_ledger_entries": n_ledger}
    bench.write_json(ctx.path("metrics.json"), metrics)
    ctx.register_artifact(ctx.path("metrics.json"), "metrics", "Aggregate audit metrics.")

    claims = build_claims(ctx, bundle, domain, behavioral)
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
    write_summary(ctx, bundle, domain, behavioral, claims, n_ledger)
    print(f"[lab11] wrote audit_report.md, ledger_reconciliation.md, "
          f"safety_case_and_rebuttal.md, and {len(claims)} drafted claims")
