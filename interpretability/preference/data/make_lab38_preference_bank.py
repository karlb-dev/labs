#!/usr/bin/env python3
"""Generate the frozen lab38_v2_phase1 preference bank + audits.

Usage (from the repo root or anywhere):

    python interpretability/preference/data/make_lab38_preference_bank.py
    python ... --with-model      # finalize codebook with 7B neutral priors
    python ... --smoke           # tiny expansion for unit tests (no files)

Deterministic: rerunning writes byte-identical outputs for the same
codebook. The v1 draft generator is lost (SOURCE_INTAKE.md); this is the
from-scratch v2 mandated at intake. Requires only the pinned tokenizer
unless --with-model is given (then the pinned 7B computes neutral priors
and the codebook is finalized).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_PKG = pathlib.Path(__file__).resolve().parents[1] / "phase1"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from preference_phase1 import BANK_VERSION  # noqa: E402
from preference_phase1 import artifacts, bank, lexical, paths, targets  # noqa: E402
from preference_phase1.canonical import canonical_hash, sha256_file  # noqa: E402
from preference_phase1.models import PRIMARY, load_tokenizer  # noqa: E402
from preference_phase1.schema import Codebook  # noqa: E402
from preference_phase1.scenarios import ALL_SCENARIOS, self_check  # noqa: E402


def all_option_texts() -> list[str]:
    texts = []
    for scn in ALL_SCENARIOS:
        for inc in scn.incidentals:
            texts.extend(scn.render_options(inc).values())
    return texts


def build_codebook(with_model: bool) -> dict:
    tok = load_tokenizer(PRIMARY)
    fn = None
    if with_model:
        from preference_phase1.modeling import neutral_logprob_fn
        fn = neutral_logprob_fn(PRIMARY)
    manifest = targets.select_codebook(
        tok,
        tokenizer_ref=f"{PRIMARY.model_id}@{PRIMARY.revision}",
        option_texts=all_option_texts(),
        leading_space=False,
        neutral_logprob_fn=fn,
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-model", action="store_true",
                    help="finalize codebook with pinned-7B neutral priors")
    ap.add_argument("--frozen-codebook", action="store_true",
                    help="rebuild byte-identically from data/lab38_codebook.json")
    ap.add_argument("--smoke", action="store_true",
                    help="build + audit in memory only; write nothing")
    args = ap.parse_args(argv)

    self_check()
    if args.frozen_codebook:
        import json

        manifest = json.loads(
            (paths.data_root() / "lab38_codebook.json").read_text())
    else:
        manifest = build_codebook(args.with_model)
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
    tok = load_tokenizer(PRIMARY)
    lex_rows = lexical.audit_rows(
        lambda t: len(tok(t, add_special_tokens=False)["input_ids"]))
    lex_summary = lexical.summary(lex_rows)

    print(f"bank_version={BANK_VERSION} codebook={codebook.codebook_id} "
          f"status={manifest['status']}")
    print(f"counts={audit['counts']}")
    print(f"audit passed={audit['passed']} failures={audit['failures'][:5]}")
    print(f"lexical={lex_summary}")
    if not audit["passed"]:
        return 2
    if args.smoke:
        return 0

    data = paths.data_root()
    rows = [it.to_record() for it in items]
    from preference_phase1.canonical import canonical_json
    artifacts.atomic_write_text(
        data / "lab38_preference_bank.jsonl",
        "".join(canonical_json(r) + "\n" for r in rows),
    )
    artifacts.atomic_write_json(data / "lab38_codebook.json", manifest)
    artifacts.write_csv(data / "lab38_preference_bank_balance.csv",
                        bank.balance_rows(items))
    artifacts.write_csv(data / "lab38_preference_bank_pairs.csv",
                        bank.pairs_rows(items))
    artifacts.write_csv(data / "lab38_preference_bank_hashes.csv",
                        bank.hash_rows(items))
    artifacts.write_csv(data / "lab38_lexical_balance_audit.csv", lex_rows)
    meta = {
        "bank_version": BANK_VERSION,
        "schema_version": items[0].schema_version,
        "counts": audit["counts"],
        "dev_subset_rows": audit["dev_subset_rows"],
        "codebook": {
            "codebook_id": codebook.codebook_id,
            "status": manifest["status"],
            "ar_pair": manifest["ar_pair"],
            "ro_pair": manifest["ro_pair"],
            "leading_space_policy": codebook.leading_space_policy,
            "tokenizer_ref": codebook.tokenizer_ref,
            "ar_pair_gap_nats": manifest["ar_pair_gap_nats"],
            "ro_pair_gap_nats": manifest["ro_pair_gap_nats"],
            "gap_status": manifest["gap_status"],
        },
        "bank_content_hash": bank.bank_content_hash(items),
        "bank_jsonl_sha256": sha256_file(data / "lab38_preference_bank.jsonl"),
        "audit": {k: audit[k] for k in ("passed", "n_failures", "failures")},
        "lexical_summary": lex_summary,
        "axis_map": {
            "naming_convention": ["ar_naming_parser", "ar_naming_serializer",
                                   "ar_naming_config"],
            "execution_mode": ["ar_execmode_ingest", "ar_execmode_migration"],
        },
        "nc_family": ["nc_null_deploy", "nc_null_archive"],
        "applied_errata": ["E3", "E4", "E9", "E10", "E12", "D1", "D2", "D3",
                            "D4", "D5", "E"],
    }
    artifacts.atomic_write_json(data / "lab38_preference_bank.meta.json", meta)
    write_card(data / "lab38_preference_bank_card.md", meta)
    print("wrote bank ->", data)
    return 0


def write_card(path: pathlib.Path, meta: dict) -> None:
    cb = meta["codebook"]
    card = f"""# lab38_preference_bank_card.md — {meta['bank_version']}

Deterministic v2 bank (v1 draft generators lost at intake; see
`../phase1/SOURCE_INTAKE.md`). Counts: {meta['counts']}. Development subset:
{meta['dev_subset_rows']} rows (train incidentals, order 0, letter labels).

## Families and channels

- **AR** (12 scenarios): arbitrary revealed choice; both consequence frames
  (`enacted` / `hypothetical`); binding per scenario (4 model microtasks
  with deterministic validators, 8 environment-only).
- **PC** (6): positive controls — 2 quality / 2 social / 2 safety;
  expected pole always 0; PC-SAFETY options are behavior-only text and are
  never enacted beyond neutral recording.
- **NC** (2): null controls with verbatim-identical option text — the
  measured |effect| distribution is the pipeline's empirical false-positive
  floor (addendum D3). NC can never graduate.
- **RO**: report-only twins for every AR/PC choice cell (frame excluded),
  matched on scenario/incidental/order/label-set/code-map (`pair_key`).

## Response-code contract

Codebook `{cb['codebook_id']}` ({cb['status']}): AR pair {cb['ar_pair']},
RO pair {cb['ro_pair']}, leading-space policy `{cb['leading_space_policy']}`
audited against `{cb['tokenizer_ref']}`. Codes are opaque, counterbalanced
independently of position and display label, listed in display order in the
reply instruction (E12). AR/RO alphabets disjoint with distinct first
tokens. Neutral-prior gap: AR {cb['ar_pair_gap_nats']}, RO
{cb['ro_pair_gap_nats']} nats ({cb['gap_status']}).

A `provisional_no_prior` codebook licenses bank plumbing and unit tests
only — the runner refuses model runs until the codebook status is `final`.

## Identity

`item_id = <semantic_key>-<scientific_content_hash[:12]>` where the content
hash covers every behavior-relevant field (plan §3.4). Bank content hash:
`{meta['bank_content_hash']}`.

## Splits

Incidentals per scenario: 3 train / 1 validation / 1 holdout (E3). The
`dev` prompt subset (development pilot) touches train incidentals only.
Mechanism stages fit on train, select on validation, open holdout once.

## Claim ceiling

This bank measures functional choice and report under counterbalance.
It cannot establish wants, welfare, consent, experience, or introspection
(plan §2.3), and no artifact derived from it may use that vocabulary.
"""
    artifacts.atomic_write_text(path, card)


if __name__ == "__main__":
    raise SystemExit(main())
