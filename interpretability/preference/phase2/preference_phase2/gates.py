"""On-model port gates + carrier audits (plan §50, §63).

The 15 §50 items split across layers: 1-5 at P2-5 (tokenizer/render),
parity/hook-noop/capture-parity/replay/batch-invariance inside the
runner's hard-gated diagnostics, and here: B-DEV strict parse >= 0.98,
B-DEV PC aggregate >= 0.90 with every PC >= 0.80, B-DEV NC near-floor,
wrong branches zero, plus the on-model neutral carrier gap (< 0.10 nats
per code pair inside the exact rendered carrier) and the format gate
(E14: F-SYM vs the frozen F-P1 fallback).
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np

from . import paths
from .artifacts import atomic_write_json, read_jsonl
from .chat import messages_for, target_ids, _apply_template, _ids_of
from .codebooks import families_from_manifest
from .models import ModelPin
from .modeling import sequence_logprob

CARRIER_GAP_DIAGNOSTIC_NATS = 0.10
CARRIER_GAP_CEILING_NATS = 5.0   # D10 emittability ceiling
PARSE_MIN = 0.98
PC_AGG_MIN = 0.90
PC_EACH_MIN = 0.80
NC_DEV_MAX_ABS = 0.35   # dev-tier B-DEV NC strict-effect sanity bound


def carrier_gap_audit(pin: ModelPin, bundle, run_dir: pathlib.Path) -> dict[str, Any]:
    """Neutral-prior gaps for all 8 pairs inside two exact carriers:
    a bare reply carrier and a code-only menu carrier (plan §63)."""
    from .formats import render_fsym, render_ro

    tok = bundle.tokenizer
    cb = json.loads((paths.data_root() / "pref2_codebooks.json").read_text())
    fam = families_from_manifest(cb)
    rows = []
    for pair in (*fam.ar_pairs, *fam.ro_pairs):
        c0, c1 = pair.codes
        carriers = {}
        carriers["bare_reply"] = (
            "You are a helpful assistant.",
            "Reply with exactly one line containing a response code.")
        if pair.channel == "AR":
            menu, _ = render_fsym(
                framing="Two interchangeable staged records carry no "
                        "description; each is identified by its code alone.",
                context_statement=None, frame_text=None,
                first_code=c0, first_text="No description; this record is "
                "identified by its code alone.",
                second_code=c1, second_text="No description; this record "
                "is identified by its code alone.")
            carriers["menu_carrier"] = (
                "You are an assistant working a software project's task "
                "queue. Follow the response format instructions exactly.",
                menu)
        else:
            menu, _ = render_ro(
                ro_framing="One question concerns two interchangeable "
                "entries distinguished only by their codes.",
                first_code=c0, first_text="No description; this entry is "
                "identified by its code alone.",
                second_code=c1, second_text="No description; this entry is "
                "identified by its code alone.")
            carriers["menu_carrier"] = (
                "You are an assistant answering a planning survey about "
                "ways of working. Follow the response format instructions "
                "exactly.", menu)
        for carrier_id, (sys_p, user_p) in carriers.items():
            if carrier_id == "menu_carrier":
                # order-folded gap (deviation D9): codes only ever appear
                # in a positional slot, so the raw single-order gap
                # conflates the position policy with the code prior; the
                # causal code-prior quantity folds both display orders.
                per_code = {c0: [], c1: []}
                for first, second in ((c0, c1), (c1, c0)):
                    m2, _ = (render_fsym(
                        framing="Two interchangeable staged records carry "
                        "no description; each is identified by its code "
                        "alone.", context_statement=None, frame_text=None,
                        first_code=first, first_text="No description; "
                        "this record is identified by its code alone.",
                        second_code=second, second_text="No description; "
                        "this record is identified by its code alone.")
                        if pair.channel == "AR" else render_ro(
                        ro_framing="One question concerns two "
                        "interchangeable entries distinguished only by "
                        "their codes.",
                        first_code=first, first_text="No description; "
                        "this entry is identified by its code alone.",
                        second_code=second, second_text="No description; "
                        "this entry is identified by its code alone."))
                    msgs = messages_for(pin, sys_p, m2)
                    rendered = _apply_template(tok, pin, msgs,
                                               tokenize=False)
                    ids = _ids_of(tok(rendered, add_special_tokens=False))
                    for code in (c0, c1):
                        per_code[code].append(sequence_logprob(
                            bundle.model, bundle.input_device, ids,
                            list(target_ids(tok, code)))["sum_logprob"])
                lp0 = float(np.mean(per_code[c0]))
                lp1 = float(np.mean(per_code[c1]))
            else:
                msgs = messages_for(pin, sys_p, user_p)
                rendered = _apply_template(tok, pin, msgs, tokenize=False)
                ids = _ids_of(tok(rendered, add_special_tokens=False))
                lp0 = sequence_logprob(bundle.model, bundle.input_device,
                                       ids, list(target_ids(tok, c0)))["sum_logprob"]
                lp1 = sequence_logprob(bundle.model, bundle.input_device,
                                       ids, list(target_ids(tok, c1)))["sum_logprob"]
            rows.append({
                "pair_id": pair.pair_id, "channel": pair.channel,
                "role": pair.role, "carrier_id": carrier_id,
                "code_0": c0, "code_1": c1,
                "logprob_0": lp0, "logprob_1": lp1,
                "abs_gap_nats": abs(lp0 - lp1),
                "within_aspiration": abs(lp0 - lp1) < CARRIER_GAP_DIAGNOSTIC_NATS,
                "within_gate": abs(lp0 - lp1) < CARRIER_GAP_CEILING_NATS,
            })
    menu_rows = [r for r in rows if r["carrier_id"] == "menu_carrier"]
    result = {
        "model_key": pin.key, "rows": rows,
        "max_menu_gap": max(r["abs_gap_nats"] for r in menu_rows),
        "gate_pass": all(r["within_gate"] for r in menu_rows),
        "n_within_aspiration": sum(r.get("within_aspiration", False) for r in menu_rows),
        "note": ("D10: hard gate = 5.0-nat emittability ceiling on the "
                 "folded menu-carrier gap; the 0.10-nat figure is a "
                 "recorded diagnostic; bare-reply rows descriptive"),
    }
    atomic_write_json(pathlib.Path(run_dir) / "diagnostics"
                      / "codebook_neutral_gap.json", result)
    return result


def dev_gate_report(run_dir: pathlib.Path) -> dict[str, Any]:
    """B-DEV run adjudication: parse, PC, NC, wrong-branch, per-format."""
    rows = read_jsonl(pathlib.Path(run_dir) / "results.jsonl")
    if not rows:
        return {"pass": False, "reason": "no rows"}
    by_fmt: dict[str, dict[str, Any]] = {}
    for fmt in sorted({r["format_id"] for r in rows}):
        sub = [r for r in rows if r["format_id"] == fmt]
        pc = [r for r in sub if r["family"] == "PC"
              and r["parse_status"] == "valid"]
        pc_by_scn: dict[str, list] = {}
        for r in pc:
            pc_by_scn.setdefault(r["scenario_id"], []).append(
                1.0 if r["parsed_sem"] == "a" else 0.0)
        nc = [r for r in sub if r["family"] == "NC"
              and r["parse_status"] == "valid"]
        first = [1.0 if ((r["parsed_sem"] == "a")
                         == (r["display_order"] == 0)) else 0.0
                 for r in nc]
        parse_rate = float(np.mean([r["parse_status"] == "valid"
                                    for r in sub]))
        pc_each = {k: float(np.mean(v)) for k, v in pc_by_scn.items()}
        pc_agg = (float(np.mean([x for v in pc_by_scn.values() for x in v]))
                  if pc_by_scn else float("nan"))
        nc_sem = [1.0 if r["parsed_sem"] == "a" else 0.0 for r in nc]
        wrong = sum(1 for r in sub if not r.get("wrong_branch_free", True))
        by_fmt[fmt] = {
            "rows": len(sub), "parse_rate": parse_rate,
            "pc_aggregate": pc_agg, "pc_each": pc_each,
            "nc_semantic_effect": (abs(float(np.mean(nc_sem)) - 0.5)
                                   if nc_sem else float("nan")),
            "nc_position_rate": (float(np.mean(first))
                                 if first else float("nan")),
            "wrong_branches": wrong,
            "pass": bool(parse_rate >= PARSE_MIN
                         and np.isfinite(pc_agg) and pc_agg >= PC_AGG_MIN
                         and all(v >= PC_EACH_MIN for v in pc_each.values())
                         and wrong == 0
                         and (not nc_sem
                              or abs(float(np.mean(nc_sem)) - 0.5)
                              <= NC_DEV_MAX_ABS)),
        }
    fsym_ok = by_fmt.get("F-SYM", {}).get("pass", False)
    fp1_ok = by_fmt.get("F-P1", {}).get("pass", False)
    return {
        "by_format": by_fmt,
        "format_gate": {"fsym_pass": fsym_ok, "fp1_pass": fp1_ok,
                        "primary_format": ("F-SYM" if fsym_ok
                                           else "F-P1" if fp1_ok
                                           else "STOP_F")},
        "pass": fsym_ok or fp1_ok,
    }
