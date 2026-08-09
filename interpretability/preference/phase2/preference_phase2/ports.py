"""Per-model tokenizer/render port audits (plan §50 items 1-5; P2-5 runs
tokenizer+render only — scientific weights stay unloaded).

Audited per model: codebook token survival (equal counts, distinct
first/final tokens, bare + space-led), exact chat-template render parity,
site token maps (strictly increasing, sentinel token-sequence constant
across rows — E6), full target tokenization, think-span absence (Qwen),
no-system shim render (Gemma).
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from . import paths
from .chat import render_item_prompt, target_ids
from .codebooks import families_from_manifest
from .models import ModelPin, chat_template_hash, load_tokenizer

SITES_AR = ("context_end", "option_a_end", "option_b_end", "menu_end",
            "response_instruction_start", "final_prompt_token")
SITES_RO = ("ro_context_end", "ro_option_a_end", "ro_option_b_end",
            "ro_menu_end", "ro_response_start", "ro_final_prompt_token")


def _bank_sample(rows: list[dict[str, Any]], *, per_bucket: int = 4) -> list[dict]:
    """Deterministic covering sample: first rows per (bank, channel,
    format, display_order) bucket."""
    buckets: dict[tuple, int] = Counter()
    out = []
    for r in rows:
        key = (r["bank"], r["channel"], r["format_id"], r["display_order"])
        if buckets[key] < per_bucket:
            buckets[key] += 1
            out.append(r)
    return out


def port_audit(pin: ModelPin, bank_rows: list[dict[str, Any]],
               *, sample_per_bucket: int = 4) -> dict[str, Any]:
    tokenizer = load_tokenizer(pin)
    cb = json.loads((paths.data_root() / "pref2_codebooks.json").read_text())
    fam = families_from_manifest(cb)

    failures: list[str] = []

    # 1. codebook token audit (bare + space-led)
    code_rows = []
    for pair in (*fam.ar_pairs, *fam.ro_pairs):
        audits = {}
        for lead in (False, True):
            pcs = []
            for code in pair.codes:
                ids = target_ids(tokenizer, (" " + code) if lead else code)
                pcs.append(ids)
            audits["space" if lead else "bare"] = pcs
        bare = audits["bare"]
        ok = (len(bare[0]) == len(bare[1])
              and bare[0][0] != bare[1][0]
              and bare[0][-1] != bare[1][-1])
        if not ok:
            failures.append(f"codebook pair {pair.pair_id} infeasible")
        code_rows.append({
            "pair_id": pair.pair_id, "channel": pair.channel,
            "role": pair.role, "codes": list(pair.codes),
            "bare_token_counts": [len(x) for x in bare],
            "space_token_counts": [len(x) for x in audits["space"]],
            "distinct_first": bare[0][0] != bare[1][0],
            "distinct_final": bare[0][-1] != bare[1][-1],
            "equal_count": len(bare[0]) == len(bare[1]),
        })

    # 2-5. render + site-map + sentinel constancy + target audit
    sample = _bank_sample(bank_rows, per_bucket=sample_per_bucket)
    parity_bad = 0
    site_bad = 0
    sentinel_seqs: dict[str, set[tuple[int, ...]]] = {"AR": set(),
                                                      "RO": set()}
    render_rows = []
    for r in sample:
        rp = render_item_prompt(tokenizer, pin, r)
        if not rp.parity_ok:
            parity_bad += 1
        sites = rp.site_token_index
        expected = SITES_RO if r["channel"] == "RO" else (
            SITES_AR if r["format_id"] == "F-SYM"
            else ("context_end", "menu_end", "response_instruction_start",
                  "final_prompt_token"))
        missing = [s for s in expected if s not in sites]
        # DISPLAY-order monotonicity: context < first-record end <
        # second-record end <= menu_end < instruction < final token.
        # Semantic a/b sites swap with display_order by design; menu_end
        # coincides with the second record's final token by construction.
        pre = "ro_" if r["channel"] == "RO" else ""
        ok = not missing
        if ok and f"{pre}option_a_end" in sites:
            a_e, b_e = sites[f"{pre}option_a_end"], sites[f"{pre}option_b_end"]
            first_e, second_e = ((a_e, b_e) if r["display_order"] == 0
                                 else (b_e, a_e))
            ctx = sites[f"{pre}context_end"]
            menu = sites[f"{pre}menu_end"]
            instr = sites["ro_response_start" if pre
                          else "response_instruction_start"]
            final = sites[f"{pre}final_prompt_token"
                          if pre else "final_prompt_token"]
            ok = (ctx < first_e < second_e <= menu < instr < final)
        elif ok:
            seq = [sites[s] for s in expected if s in sites]
            ok = seq == sorted(seq) and len(set(seq)) == len(seq)
        if not ok:
            site_bad += 1
        # sentinel token-sequence constancy (E6): tokens of the sentinel
        # line ending at context_end
        key = "RO" if r["channel"] == "RO" else "AR"
        sent = "Survey context complete." if key == "RO" else "Context complete."
        if r["format_id"] == "F-SYM" or key == "RO":
            end_tok = sites.get("ro_context_end" if key == "RO"
                                else "context_end")
            if end_tok is not None:
                n_sent = len(target_ids(tokenizer, sent))
                seq = rp.input_ids[max(0, end_tok - n_sent - 1): end_tok + 1]
                # constancy over the trailing sentinel tokens
                sentinel_seqs[key].add(seq[-n_sent:])
        render_rows.append({
            "item_id": r["item_id"], "bank": r["bank"],
            "channel": r["channel"], "format_id": r["format_id"],
            "parity_ok": rp.parity_ok,
            "prompt_tokens": len(rp.input_ids),
            "sites": dict(sites),
        })
    if parity_bad:
        failures.append(f"render parity failures: {parity_bad}")
    if site_bad:
        failures.append(f"site map failures: {site_bad}")
    for key, seqs in sentinel_seqs.items():
        if len(seqs) > 1:
            failures.append(f"{key} sentinel token sequence varies: "
                            f"{len(seqs)} variants")

    # full target tokenization audit: every code renders to the same ids
    # in isolation as when generated (bare form is the target)
    target_rows = []
    for pair in (*fam.ar_pairs, *fam.ro_pairs):
        for code in pair.codes:
            ids = target_ids(tokenizer, code)
            target_rows.append({"code": code, "pair_id": pair.pair_id,
                                "token_ids": list(ids),
                                "token_count": len(ids)})

    return {
        "model_key": pin.key, "model_id": pin.model_id,
        "revision": pin.revision,
        "chat_template_sha256": chat_template_hash(tokenizer),
        "tokenizer_class": type(tokenizer).__name__,
        "n_sampled_rows": len(sample),
        "codebook_pairs": code_rows,
        "render_sample": render_rows[:12],
        "sentinel_variants": {k: len(v) for k, v in sentinel_seqs.items()},
        "target_tokenization": target_rows,
        "failures": failures,
        "passed": not failures,
    }
