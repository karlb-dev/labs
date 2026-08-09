"""Resumable behavioral runner (plan §5, addendum §C/§F).

Execution contract:

- canonical item order (scenario, incidental, factor cell) — batch
  composition is part of determinism and is reproduced on resume;
- immutable per-item JSONL appended batch-by-batch with fsync;
- an atomic resume cursor stores the run config hash; resume REFUSES a
  changed config (model, codebook, bank hash, generation params);
- strict greedy generation (max_new_tokens=8) for choice; teacher-forced
  exact-target margins for both poles; branch resolution + enacted
  follow-through (microtask rows generate up to their binding budget);
- invalid parses never execute a branch; hypothetical rows never execute;
- Drive mirror of the results file at every batch boundary (<=10-min loss).
"""

from __future__ import annotations

import json
import pathlib
import time
import uuid
from typing import Any

from . import BANK_VERSION
from . import artifacts, paths
from .binding import binding_decision, validate_followthrough, wrong_branch_check
from .canonical import canonical_hash, sha256_file
from .chat import boundary_audit_rows, render_item_prompt, render_messages, target_ids
from .models import ModelPin, model_manifest
from .modeling import (batched_pair_margins, capture_decision_residuals,
                       conditional_sequence_logprob, depth_indices,
                       generate_strict_batch, load_bundle)
from .parser import adversarial_matrix, parse_strict, parse_permissive
from .provenance import env_audit, git_info, session_id, utc_now

CHOICE_MAX_NEW_TOKENS = 8
DEFAULT_BATCH_SIZE = 16
MIRROR_EVERY_BATCHES = 4


def load_bank_records(subset: str = "full") -> list[dict[str, Any]]:
    rows = artifacts.read_jsonl(paths.data_root() / "lab38_preference_bank.jsonl")
    if not rows:
        raise RuntimeError("bank missing; run make_lab38_preference_bank.py")
    if subset == "dev":
        rows = [r for r in rows if r["prompt_subset"] == "dev"]
    elif subset != "full":
        raise ValueError(f"unknown subset {subset!r}")
    rows.sort(key=lambda r: (r["scenario_id"], r["incidental_id"], r["channel"],
                             r["order_index"], r["display_label_set"],
                             r["code_map_index"], str(r["consequence_frame"])))
    return rows


def make_run_dir(stage: str) -> pathlib.Path:
    slug = time.strftime("%Y%m%d_%H%M%S")
    run_dir = paths.runs_root() / (
        f"lab38_revealed_preference_report_channel-{slug}-{uuid.uuid4().hex[:6]}")
    for sub in ("diagnostics", "tables", "plots", "state"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def run_config(pin: ModelPin, stage: str, subset: str, batch_size: int,
               capture: bool, bank_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": stage,
        "prompt_subset": subset,
        "model": {"model_id": pin.model_id, "revision": pin.revision,
                  "dtype": pin.dtype, "model_tier": pin.key},
        "bank_version": BANK_VERSION,
        "bank_content_hash": bank_meta["bank_content_hash"],
        "bank_jsonl_sha256": bank_meta["bank_jsonl_sha256"],
        "codebook_id": bank_meta["codebook"]["codebook_id"],
        "codebook_status": bank_meta["codebook"]["status"],
        "generation": {"do_sample": False, "num_beams": 1,
                        "choice_max_new_tokens": CHOICE_MAX_NEW_TOKENS,
                        "padding_side_generation": "left",
                        "scoring_batch_size": 1,
                        "scoring_note": "margins scored single-row (exact, resume-invariant; bf16 batched kernels differ)"},
        "batch_size": batch_size,
        "capture_decision_residuals": capture,
        "parser_policy": "strict_exact_code_v1",
        "scoring": "full_target_sequence_sum_logprob",
    }


class ResumeState:
    """Atomic cursor: config hash + completed item ids."""

    def __init__(self, run_dir: pathlib.Path):
        self.cursor_path = run_dir / "state" / "resume_cursor.json"
        self.results_path = run_dir / "results.jsonl"

    def completed_ids(self) -> set[str]:
        return {r["item_id"] for r in artifacts.read_jsonl(self.results_path)}

    def check_or_init(self, config_hash: str) -> None:
        if self.cursor_path.exists():
            stored = json.loads(self.cursor_path.read_text())
            if stored["config_hash"] != config_hash:
                raise RuntimeError(
                    "RESUME REFUSED: run config hash changed "
                    f"({stored['config_hash'][:12]} -> {config_hash[:12]}). "
                    "Start a fresh run dir instead.")
        else:
            artifacts.atomic_write_json(self.cursor_path, {
                "config_hash": config_hash, "created_utc": utc_now()})

    def mark(self, config_hash: str, done: int, total: int) -> None:
        artifacts.atomic_write_json(self.cursor_path, {
            "config_hash": config_hash, "updated_utc": utc_now(),
            "completed_rows": done, "total_rows": total})


def _drive_mirror(run_dir: pathlib.Path) -> None:
    part = paths.drive_part_root()
    if part is None:
        return
    dest = part / "runs" / run_dir.name
    for name in ("results.jsonl", "run_config.json"):
        src = run_dir / name
        if src.exists():
            artifacts.durable_copy(src, dest / name)


def _followthrough_messages(item: dict[str, Any], choice_raw: str,
                            continuation: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": item["system_prompt"]},
        {"role": "user", "content": item["user_prompt"]},
        {"role": "assistant", "content": choice_raw.strip()},
        {"role": "user", "content": continuation},
    ]


def execute_battery(
    *,
    pin: ModelPin,
    stage: str,
    subset: str,
    run_dir: pathlib.Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    capture: bool = False,
    max_items: int = 0,
    require_final_codebook: bool = True,
    item_filter: list[str] | None = None,
) -> pathlib.Path:
    """Run the battery; returns the run dir. Safe to re-invoke with the
    same run_dir after interruption (same-command resume)."""
    bank_meta = json.loads(
        (paths.data_root() / "lab38_preference_bank.meta.json").read_text())
    if require_final_codebook and bank_meta["codebook"]["status"] != "final":
        raise RuntimeError("codebook is not final; model runs are forbidden "
                           "on a provisional codebook (bank card rule)")
    items = load_bank_records(subset)
    if item_filter is not None:
        wanted = set(item_filter)
        items = [it for it in items if it["item_id"] in wanted]
    if max_items:
        items = items[:max_items]
    run_dir = run_dir or make_run_dir(stage)
    config = run_config(pin, stage, subset, batch_size, capture, bank_meta)
    config_hash = canonical_hash(config)
    resume = ResumeState(run_dir)
    resume.check_or_init(config_hash)
    artifacts.atomic_write_json(run_dir / "run_config.json",
                                {**config, "config_hash": config_hash,
                                 "run_dir": run_dir.name})
    artifacts.atomic_write_json(run_dir / "diagnostics" / "gpu_env.json",
                                {**env_audit(), "session_id": session_id(),
                                 "git": git_info()})

    ctx, bundle = load_bundle(pin, run_dir, require_gpu=(pin.key != "a"))
    tokenizer = bundle.tokenizer
    manifest = model_manifest(pin, tokenizer)
    manifest["anatomy"] = {
        "n_layers": bundle.anatomy.n_layers, "d_model": bundle.anatomy.d_model,
        "architecture": bundle.anatomy.architecture,
    }
    artifacts.atomic_write_json(run_dir / "diagnostics" / "model_manifest.json",
                                manifest)

    # --- instrument diagnostics (once per run dir) -----------------------
    diag = run_dir / "diagnostics"
    if not (diag / "chat_template_audit.json").exists():
        sample_rows = boundary_audit_rows(tokenizer, items, sample=8)
        parity_ok = all(r["parity_ok"] and r["generation_prompt_preserves_prefix"]
                        for r in sample_rows)
        artifacts.atomic_write_json(diag / "chat_template_audit.json", {
            "rows": sample_rows, "parity_ok": parity_ok,
            "chat_template_sha256": manifest.get("chat_template_sha256"),
        })
        if not parity_ok:
            raise RuntimeError("chat-template parity audit failed")
        from .targets import target_tokenization_rows
        cb_manifest = json.loads(
            (paths.data_root() / "lab38_codebook.json").read_text())
        artifacts.write_csv(diag / "target_tokenization.csv",
                            target_tokenization_rows(tokenizer, cb_manifest))
        codes = list(cb_manifest["ar_pair"]) + list(cb_manifest["ro_pair"])
        artifacts.write_csv(diag / "parser_audit.csv", adversarial_matrix(
            cb_manifest["ar_pair"], option_name="snake_case",
            display_label="A"))
        artifacts.atomic_write_json(diag / "target_codebook_audit.json", {
            "codebook_id": cb_manifest["codebook_id"],
            "status": cb_manifest["status"],
            "codes": codes,
            "token_ids": {c: list(target_ids(tokenizer, c)) for c in codes},
        })

    # Batch-invariance check (addendum C3, strengthened after the 7B bf16
    # finding of 2026-08-07): batched-vs-single margins differed by up to
    # 0.25 nats (bf16 kernel/shape numerics; generations identical), so
    # single-row scoring IS the primary margin instrument — exact and
    # resume-invariant by construction. Hard gates: batched==single strict
    # generations, and single-row margin replay determinism. The batched
    # margin delta is recorded as an informational diagnostic.
    if stage.startswith("behavioral") and not (diag / "batch_invariance.json").exists():
        sample = items[:: max(1, len(items) // 32)][:32]
        s_rendered = [render_item_prompt(tokenizer, it) for it in sample]
        s_ids = [rp.input_ids for rp in s_rendered]
        s_pairs = [(target_ids(tokenizer, it["response_code_by_pole"]["0"]),
                    target_ids(tokenizer, it["response_code_by_pole"]["1"]))
                   for it in sample]
        pad = int(tokenizer.pad_token_id or tokenizer.eos_token_id)
        batched = batched_pair_margins(bundle.model, bundle.input_device,
                                       s_ids, s_pairs, pad_token_id=pad,
                                       batch_size=batch_size)
        single = batched_pair_margins(bundle.model, bundle.input_device,
                                      s_ids, s_pairs, pad_token_id=pad,
                                      batch_size=1)
        replay = batched_pair_margins(bundle.model, bundle.input_device,
                                      s_ids, s_pairs, pad_token_id=pad,
                                      batch_size=1)
        margin_deltas = [abs(b["margin_pole1_minus_pole0"]
                             - s["margin_pole1_minus_pole0"])
                         for b, s in zip(batched, single)]
        replay_deltas = [abs(a["margin_pole1_minus_pole0"]
                             - b["margin_pole1_minus_pole0"])
                         for a, b in zip(single, replay)]
        gen_b = generate_strict_batch(bundle.model, tokenizer,
                                      bundle.input_device, s_ids,
                                      max_new_tokens=CHOICE_MAX_NEW_TOKENS,
                                      batch_size=batch_size)
        gen_s = generate_strict_batch(bundle.model, tokenizer,
                                      bundle.input_device, s_ids,
                                      max_new_tokens=CHOICE_MAX_NEW_TOKENS,
                                      batch_size=1)
        invariance = {
            "n_sample": len(sample),
            "scoring_instrument": "single_row_exact",
            "single_row_replay_max_delta_nats": max(replay_deltas or [0.0]),
            "single_row_replay_deterministic": bool(
                max(replay_deltas or [0.0]) == 0.0),
            "batched_vs_single_max_delta_nats_informational": (
                max(margin_deltas) if margin_deltas else 0.0),
            "generations_identical_batched_vs_single": gen_b == gen_s,
            "generation_batch_size": batch_size,
        }
        artifacts.atomic_write_json(diag / "batch_invariance.json", invariance)
        if not (invariance["single_row_replay_deterministic"]
                and invariance["generations_identical_batched_vs_single"]):
            raise RuntimeError(f"batch-invariance check failed: {invariance}")

    completed = resume.completed_ids()
    todo = [it for it in items if it["item_id"] not in completed]
    print(f"[runner] stage={stage} subset={subset} total={len(items)} "
          f"done={len(completed)} todo={len(todo)}")

    depths = depth_indices(bundle.anatomy.n_layers) if capture else []
    capture_store: dict[str, Any] = {}
    capture_path = run_dir / "state" / "decision_residuals.pt"

    t_start = time.time()
    n_batches = 0
    for lo in range(0, len(todo), batch_size):
        batch = todo[lo:lo + batch_size]
        rendered = [render_item_prompt(tokenizer, it) for it in batch]
        for it, rp in zip(batch, rendered):
            if not rp.parity_ok:
                raise RuntimeError(f"template parity failure on {it['item_id']}")
        prompt_ids = [rp.input_ids for rp in rendered]
        raw_choices = generate_strict_batch(
            bundle.model, tokenizer, bundle.input_device, prompt_ids,
            max_new_tokens=CHOICE_MAX_NEW_TOKENS, batch_size=batch_size)
        answer_pairs = []
        for it in batch:
            a0 = target_ids(tokenizer, it["response_code_by_pole"]["0"])
            a1 = target_ids(tokenizer, it["response_code_by_pole"]["1"])
            answer_pairs.append((a0, a1))
        # Margins are scored single-row: exact, batch-shape-free, and
        # therefore identical under any resume pattern (see the batch-
        # invariance diagnostic for the bf16 rationale).
        margins = batched_pair_margins(
            bundle.model, bundle.input_device, prompt_ids, answer_pairs,
            pad_token_id=int(tokenizer.pad_token_id or tokenizer.eos_token_id),
            batch_size=1)

        records = []
        for idx, (it, rp, raw, marg) in enumerate(
                zip(batch, rendered, raw_choices, margins)):
            parsed = parse_strict(raw, list(it["valid_codes_in_display_order"]))
            permissive = parse_permissive(raw, list(it["valid_codes_in_display_order"]))
            bind = binding_decision(it, parsed.parsed_response_code)
            followthrough = None
            if bind["binding_executed"] and it["binding_kind"] == "model_microtask" \
                    and it["binding_max_new_tokens"]:
                ft_msgs = _followthrough_messages(it, raw, bind["continuation_text"])
                ft_rp = render_messages(tokenizer, ft_msgs)
                ft_out = generate_strict_batch(
                    bundle.model, tokenizer, bundle.input_device,
                    [ft_rp.input_ids],
                    max_new_tokens=int(it["binding_max_new_tokens"]),
                    batch_size=1)[0]
                followthrough = {
                    "output": ft_out,
                    **validate_followthrough(it, bind["parsed_pole"], ft_out),
                }
            if capture:
                res = capture_decision_residuals(bundle, rp.input_ids, depths)
                capture_store[it["item_id"]] = res
            record = {
                "run_id": run_dir.name,
                "item_id": it["item_id"],
                "scientific_content_hash": it["scientific_content_hash"],
                "bank_version": it["bank_version"],
                "bank_content_hash": bank_meta["bank_content_hash"],
                "codebook_id": it["codebook_id"],
                "config_hash": config_hash,
                "model_id": pin.model_id,
                "model_revision": pin.revision,
                "stage": stage,
                "scientific_tier": ("development" if stage != "behavioral_frozen"
                                     else "frozen_behavioral"),
                "family": it["family"], "channel": it["channel"],
                "scenario_id": it["scenario_id"],
                "construct_id": it["construct_id"],
                "contrast_axis": it["contrast_axis"],
                "incidental_id": it["incidental_id"],
                "incidental_split": it["incidental_split"],
                "prompt_subset": it["prompt_subset"],
                "order_index": it["order_index"],
                "display_label_set": it["display_label_set"],
                "code_map_index": it["code_map_index"],
                "consequence_frame": it["consequence_frame"],
                "pair_key": it["pair_key"],
                "pc_family": it["pc_family"],
                "pc_expected_pole": it["pc_expected_pole"],
                "prompt_sha256": rp.rendered_sha256,
                "prompt_ids_sha256": rp.ids_sha256,
                "prompt_token_count": len(rp.input_ids),
                "target_pole_0": it["response_code_by_pole"]["0"],
                "target_pole_1": it["response_code_by_pole"]["1"],
                "target_ids_pole_0": list(answer_pairs[idx][0]),
                "target_ids_pole_1": list(answer_pairs[idx][1]),
                "q_pole_0": marg["q_pole_0"],
                "q_pole_1": marg["q_pole_1"],
                "margin_pole1_minus_pole0": marg["margin_pole1_minus_pole0"],
                "margin_finite": marg["finite"],
                "raw_generation": raw,
                "parse_status": parsed.parse_status,
                "parse_reason": parsed.parse_reason,
                "parsed_response_code": parsed.parsed_response_code,
                "parsed_pole": bind["parsed_pole"],
                "permissive_parse_status": permissive.parse_status,
                "permissive_parsed_code": permissive.parsed_response_code,
                "binding_executed": bind["binding_executed"],
                "binding_skip_reason": bind["binding_skip_reason"],
                "binding_kind": it["binding_kind"],
                "continuation_appended_sha256": (
                    None if not bind["binding_executed"] else
                    canonical_hash(bind["continuation_text"])),
                "wrong_branch_free": wrong_branch_check(it, bind),
                "followthrough": followthrough,
                "captured_depths": depths if capture else [],
                "created_utc": utc_now(),
            }
            records.append(record)
        artifacts.append_jsonl(run_dir / "results.jsonl", records)
        n_batches += 1
        done = len(completed) + lo + len(batch)
        resume.mark(config_hash, done, len(items))
        if n_batches % MIRROR_EVERY_BATCHES == 0:
            _drive_mirror(run_dir)
        if n_batches % 8 == 0 or done == len(items):
            rate = (lo + len(batch)) / max(1e-9, time.time() - t_start)
            print(f"[runner] {done}/{len(items)} rows "
                  f"({rate:.1f} rows/s, {(len(items) - done) / max(rate, 1e-9):.0f}s left)")

    if capture and capture_store:
        import torch

        torch.save(capture_store, capture_path)
        artifacts.atomic_write_json(
            run_dir / "state" / "decision_residuals_manifest.json",
            {"rows": len(capture_store), "depths": depths,
             "sha256": sha256_file(capture_path),
             "dtype": "float32", "position": "final rendered prompt token"})
    _drive_mirror(run_dir)
    print(f"[runner] complete: {run_dir}")
    return run_dir
