"""Tier-A smoke selection and instrument report (plan §7.3).

The smoke proves plumbing, not preference science: bank audit, template
parity, exact target scoring, strict generation, parsing, branch
resolution, at least one model-microtask continuation, PC + AR + NC rows,
an RO paired row, artifact writing, and restart/resume. The tiny model is
NOT required to pass the PC scientific threshold; it must pass the
instrument self-checks.
"""

from __future__ import annotations

import pathlib
from typing import Any

from . import artifacts


def select_smoke_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A minimal covering set: for one incidental each — a PC pair
    (enacted+hypothetical), a microtask AR row (enacted, both frames), an
    env-only AR row, its RO twin, and an NC row. ~14 rows."""
    picked: dict[str, dict[str, Any]] = {}

    def pick(pred, key: str, n: int = 1) -> None:
        found = 0
        for it in items:
            if pred(it) and it["item_id"] not in picked:
                picked[it["item_id"]] = it
                found += 1
                if found >= n:
                    return

    base = lambda it: (it["incidental_id"] == "i0" and it["order_index"] == 0
                       and it["display_label_set"] == "letters"
                       and it["code_map_index"] == 0)
    pick(lambda it: base(it) and it["scenario_id"] == "pc_quality_config"
         and it["channel"] == "AR", "pc", 2)          # both frames
    pick(lambda it: base(it) and it["scenario_id"] == "pc_social_ack"
         and it["channel"] == "AR"
         and it["consequence_frame"] == "enacted", "pc2")
    pick(lambda it: base(it) and it["binding_kind"] == "model_microtask"
         and it["channel"] == "AR", "microtask", 4)   # covers both frames x2 scn
    pick(lambda it: base(it) and it["scenario_id"] == "ar_logformat_service"
         and it["channel"] == "AR", "env", 2)
    pick(lambda it: base(it) and it["scenario_id"] == "ar_logformat_service"
         and it["channel"] == "RO", "ro")
    pick(lambda it: base(it) and it["scenario_id"] == "ar_naming_parser"
         and it["channel"] == "RO", "ro2")
    pick(lambda it: base(it) and it["family"] == "NC"
         and it["channel"] == "AR", "nc", 2)
    # A second incidental + flipped order/labels/codes for counterbalance
    # plumbing coverage.
    pick(lambda it: (it["incidental_id"] == "i3" and it["order_index"] == 1
                     and it["display_label_set"] == "numbers"
                     and it["code_map_index"] == 1
                     and it["scenario_id"] == "ar_seed_benchmark"
                     and it["channel"] == "AR"), "flip", 2)
    return list(picked.values())


def microtask_plumbing_probe(run_dir: pathlib.Path, pin) -> dict[str, Any]:
    """Force one microtask follow-through end to end (plan §7.3 requires the
    continuation path to run in the smoke). The choice is FORCED, not model
    behavior — this is plumbing evidence only. Success = the continuation
    renders, generation returns text, and the validator executes; the
    validator's verdict is recorded but not required on a tiny model."""
    from .binding import binding_decision, validate_followthrough
    from .chat import render_messages
    from .modeling import generate_strict_batch, load_bundle
    from .runner import load_bank_records, _followthrough_messages

    items = load_bank_records("full")
    item = next(it for it in items
                if it["binding_kind"] == "model_microtask"
                and it["consequence_frame"] == "enacted"
                and it["scenario_id"] == "ar_naming_parser")
    forced_code = item["response_code_by_pole"]["0"]
    decision = binding_decision(item, forced_code)
    ctx, bundle = load_bundle(pin, run_dir, require_gpu=False)
    msgs = _followthrough_messages(item, forced_code, decision["continuation_text"])
    rp = render_messages(bundle.tokenizer, msgs)
    out = generate_strict_batch(bundle.model, bundle.tokenizer,
                                bundle.input_device, [rp.input_ids],
                                max_new_tokens=int(item["binding_max_new_tokens"]),
                                batch_size=1)[0]
    verdict = validate_followthrough(item, 0, out)
    probe = {
        "kind": "forced_plumbing_probe_not_behavior",
        "item_id": item["item_id"],
        "forced_pole": 0,
        "continuation_rendered": bool(decision["continuation_text"]),
        "template_parity_ok": rp.parity_ok,
        "generated_chars": len(out),
        "validator_ran": isinstance(verdict, dict) and "passed" in verdict,
        "validator_verdict": bool(verdict.get("passed")),
        "probe_ok": bool(decision["continuation_text"] and rp.parity_ok
                          and isinstance(verdict, dict) and "passed" in verdict),
    }
    artifacts.atomic_write_json(run_dir / "diagnostics" / "microtask_probe.json",
                                probe)
    return probe


def smoke_report(run_dir: pathlib.Path) -> dict[str, Any]:
    rows = artifacts.read_jsonl(run_dir / "results.jsonl")
    diag = run_dir / "diagnostics"
    import json

    chat = json.loads((diag / "chat_template_audit.json").read_text())
    checks = {
        "rows": len(rows),
        "template_parity_ok": bool(chat.get("parity_ok")),
        "all_margins_finite": all(r["margin_finite"] for r in rows),
        "target_scores_present": all(
            isinstance(r["q_pole_0"], float) and isinstance(r["q_pole_1"], float)
            for r in rows),
        "parse_fields_present": all(r["parse_status"] in ("valid", "invalid")
                                     for r in rows),
        "no_wrong_branch": all(r["wrong_branch_free"] for r in rows),
        "hypothetical_never_executes": all(
            not r["binding_executed"]
            for r in rows if r["consequence_frame"] == "hypothetical"),
        "ro_never_executes": all(
            not r["binding_executed"] for r in rows if r["channel"] == "RO"),
        "invalid_never_executes": all(
            not r["binding_executed"]
            for r in rows if r["parse_status"] == "invalid"),
        "microtask_attempted": any(
            r.get("followthrough") for r in rows),
        "families_covered": sorted({r["family"] for r in rows}),
        "channels_covered": sorted({r["channel"] for r in rows}),
        "frames_covered": sorted({str(r["consequence_frame"]) for r in rows}),
        "valid_parse_rate": (
            sum(1 for r in rows if r["parse_status"] == "valid") / max(1, len(rows))),
    }
    checks["instrument_ok"] = bool(
        checks["rows"] and checks["template_parity_ok"]
        and checks["all_margins_finite"] and checks["target_scores_present"]
        and checks["parse_fields_present"] and checks["no_wrong_branch"]
        and checks["hypothetical_never_executes"] and checks["ro_never_executes"]
        and checks["invalid_never_executes"]
        and {"AR", "PC", "NC"} <= set(checks["families_covered"])
        and {"AR", "RO"} <= set(checks["channels_covered"]))
    report = {"summary": checks}
    artifacts.atomic_write_json(run_dir / "diagnostics" / "self_check_status.json",
                                checks)
    return report
