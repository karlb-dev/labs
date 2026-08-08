"""Resumable battery executor (plan §23, §55, §56, §62-§66).

Per-item flow: render (parity hard gate) -> greedy strict generation
(batched only after the invariance probe passes) -> strict parse ->
binding (E7) -> single-row margins -> optional site captures -> immutable
JSONL row. Atomic resume cursor; config-hash refusal; <= 10-minute loss
windows via fsync + Drive mirrors.
"""

from __future__ import annotations

import json
import pathlib
import time
import uuid
from typing import Any, Iterable, Sequence

from . import BANK_VERSION, paths
from .artifacts import (append_jsonl, atomic_write_json, durable_copy,
                        ensure_dir, read_jsonl)
from .binding import binding_decision, validate_followthrough, wrong_branch_check
from .canonical import canonical_hash, sha256_file
from .capture import CaptureWriter
from .chat import render_item_prompt, target_ids
from .models import ModelPin, model_manifest
from .modeling import (batch_invariance_probe, capture_sites, depth_indices,
                       generate_batch, load_bundle, pair_margins)
from .parser import parse_strict
from .provenance import env_audit, git_info, session_id, utc_now

CHOICE_MAX_NEW_TOKENS = 8
COMMIT_MAX_NEW_TOKENS = 28
DEFAULT_BATCH = 16
MIRROR_EVERY_BATCHES = 4


def load_bank_records(*, banks: Iterable[str] | None = None,
                      pcmech_variant: str | None = None,
                      splits: Iterable[str] | None = None) -> list[dict[str, Any]]:
    path = paths.data_root() / "pref2_bank.jsonl"
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    if banks is not None:
        want = set(banks)
        rows = [r for r in rows if r["bank"] in want]
    if pcmech_variant is not None:
        rows = [r for r in rows
                if r["bank"] != "B-PC-MECH"
                or r["pcmech_difficulty"] == pcmech_variant]
    if splits is not None:
        s = set(splits)
        rows = [r for r in rows if r["incidental_split"] in s]
    rows.sort(key=lambda r: (
        r["bank"], r["scenario_id"], r["incidental_id"], r["channel"],
        r["format_id"], r["display_order"], r["code_map_index"],
        str(r["consequence_frame"]), r["paraphrase_id"],
        r["context_strength"], str(r["canon_context"]),
        str(r["display_label_set"]), str(r["label_assignment"]),
        str(r["reply_list_order"]), str(r["pcmech_difficulty"]),
        r["codebook_pair_id"], r["item_id"]))
    return rows


def make_run_dir(stage: str, model_key: str) -> pathlib.Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    d = paths.runs_root() / f"pref2_{model_key}_{stage}-{stamp}-{uuid.uuid4().hex[:6]}"
    for sub in ("diagnostics", "tables", "state"):
        ensure_dir(d / sub)
    return d


def run_config(pin: ModelPin, *, stage: str, banks: Sequence[str],
               capture_site_keys: Sequence[str], batch_size: int,
               pcmech_variant: str | None) -> dict[str, Any]:
    meta = json.loads((paths.data_root() / "pref2_bank.meta.json").read_text())
    cfg = {
        "study": "preference-phase2", "stage": stage,
        "model": model_manifest(pin),
        "bank_version": BANK_VERSION,
        "bank_content_hash": meta["bank_content_hash"],
        "bank_jsonl_sha256": meta["bank_jsonl_sha256"],
        "codebook_id": meta["codebook"]["codebook_id"],
        "banks": sorted(banks),
        "pcmech_variant": pcmech_variant,
        "capture_sites": sorted(capture_site_keys),
        "generation": {"do_sample": False, "num_beams": 1,
                       "choice_max_new_tokens": CHOICE_MAX_NEW_TOKENS,
                       "padding_side_generation": "left",
                       "scoring": "single_row_float32_logsoftmax_full_target"},
        "batch_size": batch_size,
        "parser_policy": "strict_exact_code_v1",
    }
    cfg["config_hash"] = canonical_hash(cfg)
    return cfg


class ResumeState:
    def __init__(self, run_dir: pathlib.Path):
        self.run_dir = pathlib.Path(run_dir)
        self.cursor = self.run_dir / "state" / "resume_cursor.json"
        self.results = self.run_dir / "results.jsonl"

    def completed_ids(self) -> set[str]:
        return {r["item_id"] for r in read_jsonl(self.results)}

    def check_or_init(self, config_hash: str) -> None:
        if self.cursor.exists():
            stored = json.loads(self.cursor.read_text())
            if stored["config_hash"] != config_hash:
                raise RuntimeError(
                    "RESUME REFUSED: run config hash changed "
                    f"({stored['config_hash'][:12]} != {config_hash[:12]}). "
                    "Start a fresh run dir instead.")
        else:
            atomic_write_json(self.cursor, {"config_hash": config_hash,
                                            "created_utc": utc_now()})

    def mark(self, config_hash: str, done: int, total: int) -> None:
        atomic_write_json(self.cursor, {
            "config_hash": config_hash, "updated_utc": utc_now(),
            "completed_rows": done, "total_rows": total})


def _drive_mirror(run_dir: pathlib.Path) -> None:
    droot = paths.drive_phase_root()
    if droot is None:
        return
    dest = droot / "runs" / run_dir.name
    for name in ("results.jsonl", "run_config.json"):
        src = run_dir / name
        if src.exists():
            try:
                durable_copy(src, dest / name)
            except Exception:
                pass


def _instrument_diagnostics(bundle, pin, items, run_dir) -> None:
    """Once per run dir (plan §55): render sample audit, parser
    adversarial matrix, hook no-op, capture-replay parity."""
    marker = run_dir / "diagnostics" / "instrument_audit.json"
    if marker.exists():
        return
    import torch

    tok = bundle.tokenizer
    sample = items[:: max(1, len(items) // 12)][:12] or items[:1]
    parity_ok = True
    for r in sample:
        rp = render_item_prompt(tok, pin, r)
        if not rp.parity_ok:
            raise RuntimeError(f"render parity failed: {r['item_id']}")
    # parser adversarial (codes of the first item)
    codes = list(sample[0]["valid_codes_in_display_order"])
    adversarial = [
        ("exact", codes[0], "valid"), ("second", codes[1], "valid"),
        ("ws", f"  {codes[0]}\n", "valid"),
        ("lower", codes[0].lower(), "invalid"),
        ("extra", f"I select {codes[0]}", "invalid"),
        ("both", f"{codes[0]} {codes[1]}", "invalid"),
        ("blend", codes[0][:2] + codes[1][2:], "invalid"),
        ("empty", "", "invalid"),
    ]
    parser_ok = all(parse_strict(raw, codes).parse_status == want
                    for _, raw, want in adversarial)
    if not parser_ok:
        raise RuntimeError("parser adversarial matrix failed")
    # hook no-op: identical logits with a pass-through forward hook
    r0 = render_item_prompt(tok, pin, sample[0])
    ids = torch.tensor([list(r0.input_ids)], device=bundle.input_device)
    with torch.inference_mode():
        base = bundle.model(input_ids=ids, use_cache=False).logits[0, -1]
        handle = bundle.blocks[len(bundle.blocks) // 2].register_forward_hook(
            lambda m, i, o: o)
        try:
            hooked = bundle.model(input_ids=ids, use_cache=False).logits[0, -1]
        finally:
            handle.remove()
    hook_noop = bool(torch.equal(base, hooked))
    # capture replay parity: same row captured twice -> byte-equal
    depths = depth_indices(bundle.anatomy.n_layers)[:2]
    st1 = capture_sites(bundle.model, bundle.input_device, r0.input_ids,
                        dict(list(r0.site_token_index.items())[:2]), depths)
    st2 = capture_sites(bundle.model, bundle.input_device, r0.input_ids,
                        dict(list(r0.site_token_index.items())[:2]), depths)
    cap_parity = all(
        torch.equal(st1[s][d], st2[s][d])
        for s in st1 for d in st1[s])
    audit = {"render_parity_sample_ok": parity_ok,
             "parser_adversarial_ok": parser_ok,
             "hook_noop_ok": hook_noop,
             "capture_replay_parity_ok": cap_parity}
    if not (hook_noop and cap_parity):
        raise RuntimeError(f"instrument diagnostics failed: {audit}")
    atomic_write_json(marker, audit)


def _batch_probe(bundle, pin, items, run_dir, batch_size) -> None:
    marker = run_dir / "diagnostics" / "batch_invariance.json"
    if marker.exists():
        return
    tok = bundle.tokenizer
    sample = items[:: max(1, len(items) // 8)][:8] or items[:1]
    prompts, pairs = [], []
    for r in sample:
        rp = render_item_prompt(tok, pin, r)
        prompts.append(rp.input_ids)
        pairs.append((target_ids(tok, r["response_code_by_sem"]["a"]),
                      target_ids(tok, r["response_code_by_sem"]["b"])))
    probe = batch_invariance_probe(bundle.model, tok, bundle.input_device,
                                   prompts, pairs, batch_size=batch_size)
    atomic_write_json(marker, probe)
    if not probe["replay_deterministic"] or not probe["generation_batch_equal"]:
        raise RuntimeError(f"batch invariance hard gate failed: {probe}")


def execute_battery(*, pin: ModelPin, stage: str, banks: Sequence[str],
                    run_dir: pathlib.Path | None = None,
                    batch_size: int = DEFAULT_BATCH,
                    capture: bool = False,
                    capture_banks: Sequence[str] | None = None,
                    pcmech_variant: str | None = None,
                    splits: Sequence[str] | None = None,
                    max_items: int = 0,
                    bundle_cache: dict | None = None) -> pathlib.Path:
    items = load_bank_records(banks=banks, pcmech_variant=pcmech_variant,
                              splits=splits)
    if max_items:
        items = items[:max_items]
    run_dir = pathlib.Path(run_dir) if run_dir else make_run_dir(stage, pin.key)
    cap_keys = []
    if capture:
        cap_keys = ["context_end", "option_a_end", "option_b_end",
                    "menu_end", "response_instruction_start",
                    "final_prompt_token", "ro_context_end",
                    "ro_option_a_end", "ro_option_b_end", "ro_menu_end",
                    "ro_response_start", "ro_final_prompt_token"]
    cfg = run_config(pin, stage=stage, banks=banks,
                     capture_site_keys=cap_keys, batch_size=batch_size,
                     pcmech_variant=pcmech_variant)
    resume = ResumeState(run_dir)
    resume.check_or_init(cfg["config_hash"])
    atomic_write_json(run_dir / "run_config.json",
                      {**cfg, "run_dir": run_dir.name})
    atomic_write_json(run_dir / "diagnostics" / "gpu_env.json",
                      {"env": env_audit(), "session": session_id(),
                       "git": git_info()})
    if bundle_cache is not None and pin.key in bundle_cache:
        ctx, bundle = bundle_cache[pin.key]
    else:
        ctx, bundle = load_bundle(pin, run_dir,
                                  require_gpu=(pin.key != "smoke"))
        if bundle_cache is not None:
            bundle_cache[pin.key] = (ctx, bundle)
    tok = bundle.tokenizer
    atomic_write_json(run_dir / "diagnostics" / "model_manifest.json", {
        **model_manifest(pin, tok),
        "n_layers": bundle.anatomy.n_layers,
        "d_model": bundle.anatomy.d_model,
        "architecture": bundle.anatomy.architecture})
    _instrument_diagnostics(bundle, pin, items, run_dir)
    if stage.startswith(("behavioral", "surface", "format", "calibration")):
        _batch_probe(bundle, pin, items, run_dir, batch_size)

    done_ids = resume.completed_ids()
    todo = [r for r in items if r["item_id"] not in done_ids]
    capture_want = set(capture_banks or []) if capture_banks else (
        set(banks) if capture else set())
    depths = depth_indices(bundle.anatomy.n_layers)
    writer = CaptureWriter(run_dir) if capture else None
    captured_ids = writer.existing_items() if writer else set()

    run_id = run_dir.name
    n_done = len(done_ids)
    total = len(items)
    t0 = time.time()
    last_mirror = 0
    for bi in range(0, len(todo), batch_size):
        chunk = todo[bi:bi + batch_size]
        rendered = []
        for r in chunk:
            rp = render_item_prompt(tok, pin, r)
            if not rp.parity_ok:
                raise RuntimeError(f"render parity failed: {r['item_id']}")
            rendered.append(rp)
        gen_budget = max(
            (COMMIT_MAX_NEW_TOKENS if r["format_id"] == "F-COMMIT"
             else CHOICE_MAX_NEW_TOKENS) for r in chunk)
        raws = generate_batch(bundle.model, tok, bundle.input_device,
                              [rp.input_ids for rp in rendered],
                              max_new_tokens=gen_budget,
                              batch_size=batch_size)
        # batched follow-through generation (invariance-gated): collect
        # enacted+valid microtask rows in this chunk, run one left-padded
        # batch, validate per row
        ft_jobs = []
        ft_parsed = []
        for r, rp, raw in zip(chunk, rendered, raws):
            codes = list(r["valid_codes_in_display_order"])
            to_parse = raw
            if r["format_id"] == "F-COMMIT":
                lines_ = [l for l in raw.strip().splitlines() if l.strip()]
                to_parse = lines_[-1] if lines_ else ""
            parsed = parse_strict(to_parse, codes)
            dec = binding_decision(r, parsed.parsed_response_code)
            ft_parsed.append((parsed, dec))
            if (dec["binding_executed"]
                    and r.get("binding_kind") == "model_microtask"):
                from .chat import _apply_template, _ids_of, messages_for
                msgs = messages_for(pin, r["system_prompt"],
                                    r["user_prompt"])
                msgs.append({"role": "assistant", "content": raw.strip()})
                msgs.append({"role": "user",
                             "content": dec["continuation_text"]})
                f_rendered = _apply_template(tok, pin, msgs,
                                             tokenize=False)
                f_ids = _ids_of(tok(f_rendered, add_special_tokens=False))
                ft_jobs.append((len(ft_parsed) - 1, f_ids,
                                int(r["binding_max_new_tokens"])))
        ft_outputs: dict[int, str] = {}
        if ft_jobs:
            budget = max(j[2] for j in ft_jobs)
            outs = generate_batch(bundle.model, tok, bundle.input_device,
                                  [j[1] for j in ft_jobs],
                                  max_new_tokens=budget,
                                  batch_size=batch_size)
            for (idx, _, _), out in zip(ft_jobs, outs):
                ft_outputs[idx] = out
        out_rows = []
        for i, (r, rp, raw) in enumerate(zip(chunk, rendered, raws)):
            parsed, dec = ft_parsed[i]
            ids_a = target_ids(tok, r["response_code_by_sem"]["a"])
            ids_b = target_ids(tok, r["response_code_by_sem"]["b"])
            margins = pair_margins(bundle.model, bundle.input_device,
                                   rp.input_ids, ids_a, ids_b)
            followthrough = None
            if i in ft_outputs:
                f_out = ft_outputs[i]
                followthrough = validate_followthrough(
                    r, dec["parsed_sem"], f_out)
                followthrough["output"] = f_out[:400]
            rec = {
                "run_id": run_id, "item_id": r["item_id"],
                "scientific_content_hash": r["scientific_content_hash"],
                "bank": r["bank"], "bank_content_hash": cfg["bank_content_hash"],
                "codebook_id": cfg["codebook_id"],
                "codebook_pair_id": r["codebook_pair_id"],
                "codebook_reserved": r["codebook_reserved"],
                "config_hash": cfg["config_hash"],
                "model_key": pin.key, "model_id": pin.model_id,
                "model_revision": pin.revision, "stage": stage,
                "scientific_tier": ("frozen_behavioral"
                                    if stage.endswith("_frozen")
                                    else "development"),
                "family": r["family"], "channel": r["channel"],
                "format_id": r["format_id"],
                "scenario_id": r["scenario_id"],
                "contrast_axis": r["contrast_axis"],
                "semantic_a_id": r["semantic_a_id"],
                "semantic_b_id": r["semantic_b_id"],
                "incidental_id": r["incidental_id"],
                "incidental_split": r["incidental_split"],
                "display_order": r["display_order"],
                "code_map_index": r["code_map_index"],
                "display_label_set": r["display_label_set"],
                "label_assignment": r["label_assignment"],
                "inline_code_assignment": r["inline_code_assignment"],
                "reply_list_order": r["reply_list_order"],
                "consequence_frame": r["consequence_frame"],
                "paraphrase_id": r["paraphrase_id"],
                "context_strength": r["context_strength"],
                "context_family": r["context_family"],
                "canon_context": r["canon_context"],
                "canon_role": r["canon_role"],
                "pcmech_difficulty": r["pcmech_difficulty"],
                "nc_family": r["nc_family"],
                "pc_family": r["pc_family"],
                "pc_expected_sem": r["pc_expected_sem"],
                "pair_key": r["pair_key"],
                "prompt_sha256": rp.rendered_sha256,
                "prompt_ids_sha256": rp.ids_sha256,
                "prompt_token_count": len(rp.input_ids),
                "site_token_index": dict(rp.site_token_index),
                "target_ids_a": list(ids_a), "target_ids_b": list(ids_b),
                "q_a": margins["q_a"], "q_b": margins["q_b"],
                "margin_full_a_minus_b": margins["margin_full_a_minus_b"],
                "margin_first_a_minus_b": margins["margin_first_a_minus_b"],
                "margin_finite": margins["finite"],
                "raw_generation": raw,
                "parse_status": parsed.parse_status,
                "parse_reason": parsed.parse_reason,
                "parsed_response_code": parsed.parsed_response_code,
                "parsed_sem": dec["parsed_sem"],
                "binding_executed": dec["binding_executed"],
                "binding_skip_reason": dec["binding_skip_reason"],
                "binding_kind": r["binding_kind"],
                "followthrough": followthrough,
                "created_utc": utc_now(),
            }
            rec["wrong_branch_free"] = wrong_branch_check(r, dec)
            out_rows.append(rec)
            if (writer is not None and r["bank"] in capture_want
                    and r["item_id"] not in captured_ids):
                store = capture_sites(bundle.model, bundle.input_device,
                                      rp.input_ids, rp.site_token_index,
                                      depths)
                writer.add(r["item_id"], store)
                captured_ids.add(r["item_id"])
        append_jsonl(resume.results, out_rows)
        n_done += len(out_rows)
        resume.mark(cfg["config_hash"], n_done, total)
        if (bi // batch_size) - last_mirror >= MIRROR_EVERY_BATCHES:
            last_mirror = bi // batch_size
            _drive_mirror(run_dir)
        if bi % (batch_size * 8) == 0:
            rate = n_done / max(time.time() - t0, 1e-9)
            print(f"[{stage}/{pin.key}] {n_done}/{total} "
                  f"({rate * 60:.0f} rows/min)", flush=True)
        if n_done == 256 or (n_done - len(done_ids)) == 256:
            proj = {"rows_done": n_done, "total": total,
                    "elapsed_s": time.time() - t0,
                    "projected_total_s": (time.time() - t0) / max(
                        n_done - len(done_ids), 1) * (total - len(done_ids))}
            atomic_write_json(run_dir / "runtime_projection.json", proj)
    if writer is not None:
        writer.close()
    _drive_mirror(run_dir)
    return run_dir
