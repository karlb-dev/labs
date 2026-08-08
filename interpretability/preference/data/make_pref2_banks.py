#!/usr/bin/env python3
"""Freeze the Phase 2 v3 banks (plan §5; addendum D).

Deterministic: reruns are byte-identical. Writes, next to this script:

    pref2_bank.jsonl        every bank row (canonical JSON lines)
    pref2_bank.meta.json    counts, hashes, codebook id, audit
    pref2_codebooks.json    codebook family manifest

Usage: python make_pref2_banks.py  (tokenizer downloads via HF hub; the
primary 32B tokenizer at its frozen revision is the selection reference).
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase2"))

PRIMARY_TOKENIZER = "allenai/Olmo-3.1-32B-Instruct"
PRIMARY_REVISION = "ac0587e4a7744a551c059d8cd17ba220bc940dae"


def main() -> int:
    from transformers import AutoTokenizer

    from preference_phase2 import banks, codebooks, scenarios
    from preference_phase2.canonical import canonical_json, sha256_file

    scenarios.self_check()
    tok = AutoTokenizer.from_pretrained(PRIMARY_TOKENIZER,
                                        revision=PRIMARY_REVISION)
    opt_texts = []
    for s in scenarios.ALL_SCENARIOS:
        for inc in s.incidentals:
            for tpl in (*s.option_templates_a, *s.option_templates_b,
                        *s.ro_option_templates_a, *s.ro_option_templates_b):
                opt_texts.append(s.render(tpl, inc))
    fam = codebooks.build_families(
        tok, tokenizer_ref=f"{PRIMARY_TOKENIZER}@{PRIMARY_REVISION[:8]}",
        option_texts=opt_texts)

    items = banks.build_bank(fam)
    audit = banks.audit_bank(items, fam)
    if not audit["passed"]:
        for f in audit["failures"]:
            print("FAIL:", f, file=sys.stderr)
        return 2

    bank_path = HERE / "pref2_bank.jsonl"
    with open(bank_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(canonical_json(it.to_record()) + "\n")

    cb_path = HERE / "pref2_codebooks.json"
    cb_path.write_text(json.dumps(fam.to_manifest(), indent=2,
                                  sort_keys=True) + "\n")

    meta = {
        "schema_version": 2,
        "bank_version": banks.BANK_VERSION,
        "bank_content_hash": banks.bank_content_hash(items),
        "bank_jsonl_sha256": sha256_file(bank_path),
        "codebook": fam.to_manifest(),
        "counts": audit["counts"],
        "total": audit["total"],
        "audit": {"passed": audit["passed"],
                  "n_failures": audit["n_failures"]},
        "primary_tokenizer": f"{PRIMARY_TOKENIZER}@{PRIMARY_REVISION}",
    }
    (HERE / "pref2_bank.meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: meta[k] for k in
                      ("bank_content_hash", "bank_jsonl_sha256", "total")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
