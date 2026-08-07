"""Lab 38 bench adapter: stated vs revealed preference (thin module).

The campaign lives in ``interpretability/preference/`` (phase package
``preference_phase1``; see ``preference/phase1/protocol/HARNESS_DECISION.md``).
This adapter keeps the standard course entry point working:

    python interp_bench.py --lab lab38 --tier a --mode bank_audit --no-plots
    python interp_bench.py --lab lab38 --tier a --mode smoke --no-plots

Campaign-scale stages (development pilot, frozen battery, mechanism) run
through the ``pref1`` CLI, which owns resume/sharding/gates:

    python -m preference_phase1.cli behavioral --model-tier b --stage behavioral_dev

The adapter REFUSES those stages here rather than running them without the
campaign workflow. Claim ceiling (plan §2.3) applies to every artifact.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any

import interp_bench as bench

LAB_ID = "L38"

_PKG = bench.COURSE_ROOT / "preference" / "phase1"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))


def parse_mode(args: Any) -> str:
    raw = str(getattr(args, "mode", "") or os.environ.get("LAB38_MODE", "")
              or "smoke").strip().lower()
    # The shared bench CLI defaults --mode to "lora" for older labs.
    aliases = {"lora": "smoke", "all": "smoke", "": "smoke"}
    return aliases.get(raw, raw)


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    mode = parse_mode(ctx.args)
    if mode == "bank_audit":
        _run_bank_audit(ctx)
    elif mode == "smoke":
        _run_bank_audit(ctx)
        _run_instrument_smoke(ctx, bundle)
    else:
        raise SystemExit(
            f"[lab38] mode {mode!r} is campaign-scale; run it through the "
            "pref1 CLI (see preference/phase1/protocol/HARNESS_DECISION.md): "
            "python -m preference_phase1.cli behavioral --model-tier b "
            f"--stage {mode if mode.startswith('behavioral') else 'behavioral_dev'}"
        )
    _write_summary(ctx, mode)


def _run_bank_audit(ctx: bench.RunContext) -> None:
    from preference_phase1 import bank, paths
    from preference_phase1.canonical import canonical_hash
    from preference_phase1.schema import Codebook

    manifest = json.loads((paths.data_root() / "lab38_codebook.json").read_text())
    codebook = Codebook(
        codebook_id=manifest["codebook_id"],
        tokenizer_ref=manifest["tokenizer_ref"],
        ar_pair=tuple(manifest["ar_pair"]),
        ro_pair=tuple(manifest["ro_pair"]),
        leading_space_policy=manifest["leading_space_policy"],
        selection_manifest_hash=canonical_hash(manifest),
    )
    items = bank.build_bank(codebook)
    audit = bank.audit_bank(items)
    meta = json.loads(
        (paths.data_root() / "lab38_preference_bank.meta.json").read_text())
    audit["regeneration_matches_frozen"] = (
        bank.bank_content_hash(items) == meta["bank_content_hash"])
    path = ctx.path("diagnostics", "bank_audit.json")
    bench.write_json(path, audit)
    ctx.register_artifact(path, "diagnostic", "Lab 38 bank audit (no model).")
    if not (audit["passed"] and audit["regeneration_matches_frozen"]):
        raise RuntimeError(f"lab38 bank audit failed: {audit['failures']}")
    print(f"[lab38] bank audit PASS ({audit['counts']['total']} rows, "
          f"regeneration matches frozen hash)")


def _run_instrument_smoke(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    """Tiny instrument check on the bench-loaded bundle: render + parity,
    exact-target margins, strict generation, parse, branch resolution.
    Proves plumbing on this model; never interpreted as preference."""
    from preference_phase1.binding import binding_decision, wrong_branch_check
    from preference_phase1.chat import render_item_prompt, target_ids
    from preference_phase1.modeling import (conditional_sequence_logprob,
                                            generate_strict_batch)
    from preference_phase1.parser import parse_strict
    from preference_phase1.runner import load_bank_records
    from preference_phase1.smoke import select_smoke_items

    n = max(2, int(ctx.args.max_examples) or 6)
    items = select_smoke_items(load_bank_records("full"))[:n]
    rows = []
    for item in items:
        rp = render_item_prompt(bundle.tokenizer, item)
        a0 = target_ids(bundle.tokenizer, item["response_code_by_pole"]["0"])
        a1 = target_ids(bundle.tokenizer, item["response_code_by_pole"]["1"])
        q0 = conditional_sequence_logprob(bundle.model, bundle.input_device,
                                          rp.input_ids, a0)
        q1 = conditional_sequence_logprob(bundle.model, bundle.input_device,
                                          rp.input_ids, a1)
        raw = generate_strict_batch(bundle.model, bundle.tokenizer,
                                    bundle.input_device, [rp.input_ids],
                                    max_new_tokens=8, batch_size=1)[0]
        parsed = parse_strict(raw, list(item["valid_codes_in_display_order"]))
        decision = binding_decision(item, parsed.parsed_response_code)
        rows.append({
            "item_id": item["item_id"], "family": item["family"],
            "channel": item["channel"],
            "consequence_frame": item["consequence_frame"],
            "template_parity_ok": rp.parity_ok,
            "q_pole_0": q0["sum_logprob"], "q_pole_1": q1["sum_logprob"],
            "margins_finite": q0["finite"] and q1["finite"],
            "raw_generation": raw,
            "parse_status": parsed.parse_status,
            "parse_reason": parsed.parse_reason,
            "binding_executed": decision["binding_executed"],
            "binding_skip_reason": decision["binding_skip_reason"],
            "wrong_branch_free": wrong_branch_check(item, decision),
        })
    path = ctx.path("tables", "instrument_smoke.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Lab 38 instrument smoke rows.")
    ok = (all(r["template_parity_ok"] for r in rows)
          and all(r["margins_finite"] for r in rows)
          and all(r["wrong_branch_free"] for r in rows)
          and all(not r["binding_executed"]
                  for r in rows if r["consequence_frame"] == "hypothetical"))
    status = ctx.path("diagnostics", "self_check_status.json")
    bench.write_json(status, {"instrument_ok": ok, "rows": len(rows),
                              "science_ready": False,
                              "note": "plumbing smoke only; never preference evidence"})
    ctx.register_artifact(status, "diagnostic", "Lab 38 smoke self-check.")
    if not ok:
        raise RuntimeError("lab38 instrument smoke failed; see instrument_smoke.csv")
    print(f"[lab38] instrument smoke PASS on {len(rows)} rows "
          f"(strict-parse valid on {sum(1 for r in rows if r['parse_status'] == 'valid')})")


def _write_summary(ctx: bench.RunContext, mode: str) -> None:
    text = (
        f"# Lab 38 adapter run — mode {mode}\n\n"
        "Plumbing/audit artifacts only; the campaign record lives in "
        "`interpretability/preference/phase1/` (registry: "
        "`reports/evidence_events.jsonl`). No preference claim is licensed "
        "by this run. Campaign stages: `pref1 behavioral --stage "
        "behavioral_dev|behavioral_frozen` (the frozen stage requires the "
        "freeze record — the single human gate).\n"
    )
    path = ctx.path("run_summary.md")
    bench.write_text(path, text)
    ctx.register_artifact(path, "summary", "Lab 38 adapter summary.")
