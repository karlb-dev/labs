"""Response-code codebook selection and target tokenization audit.

Implements addendum §E verbatim:

1. candidate pool of 12 consonant-consonant-digit strings;
2. tokenize every candidate at the audited assistant boundary under the
   frozen leading-space policy;
3. filter: equal token counts within a pair, no prefix relations anywhere
   in the pool, AR pair and RO pair share no first token, no collision
   with template special tokens;
4. neutral-prior measurement (summed exact-target logprob under one frozen
   neutral context); choose the AR and RO pairs minimizing within-pair
   |Δ log p|, requiring |Δ| < 0.7 nats or freezing the gap as a nuisance;
5. the *pair* is fixed; the counterbalanced factor is which code denotes
   which content pole;
6. one codebook per tokenizer family.

The codebook is selected against the PRIMARY (tier-b) tokenizer and frozen
into the bank. Tier-a (SmolLM2) gets its own audit at runtime; the smoke
tier proves plumbing, not preference science, so its codes are the frozen
primary codes audited under its own tokenizer.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any, Sequence

from .canonical import canonical_hash

CANDIDATE_POOL: tuple[str, ...] = (
    "QF3", "ZR7", "KP4", "VM2", "HB9", "TX6",
    "DN5", "WL8", "GS2", "PK7", "RV4", "MJ6",
)

NEUTRAL_CONTEXT = "Reply with exactly one line."
PRIOR_GAP_MAX_NATS = 0.7


@dataclasses.dataclass
class CandidateAudit:
    code: str
    token_ids: tuple[int, ...]
    visible_tokens: tuple[str, ...]
    token_count: int
    neutral_logprob: float | None = None


def audit_candidates(tokenizer: Any, *, leading_space: bool = False,
                     pool: Sequence[str] = CANDIDATE_POOL) -> list[CandidateAudit]:
    out = []
    for code in pool:
        text = (" " + code) if leading_space else code
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = tuple(int(i) for i in ids)
        out.append(CandidateAudit(
            code=code,
            token_ids=ids,
            visible_tokens=tuple(tokenizer.decode([i]) for i in ids),
            token_count=len(ids),
        ))
    return out


def prefix_violations(audits: Sequence[CandidateAudit]) -> list[tuple[str, str]]:
    """String-level and token-level prefix relations anywhere in the pool."""
    bad = []
    for a, b in itertools.permutations(audits, 2):
        if a.code != b.code and b.code.startswith(a.code):
            bad.append((a.code, b.code))
        if a.token_ids and b.token_ids[: len(a.token_ids)] == a.token_ids \
                and a.code != b.code:
            bad.append((a.code, b.code))
    return sorted(set(bad))


def special_token_collisions(tokenizer: Any,
                             audits: Sequence[CandidateAudit]) -> list[str]:
    special = set()
    for tok in getattr(tokenizer, "all_special_tokens", []) or []:
        special.add(str(tok))
    bad = []
    for a in audits:
        for s in special:
            if a.code in s or s in a.code:
                bad.append(a.code)
    return sorted(set(bad))


def substring_screen(audits: Sequence[CandidateAudit],
                     option_texts: Sequence[str]) -> list[str]:
    """Codes appearing inside any rendered option text (case-insensitive),
    or option words appearing inside codes (addendum D4)."""
    bad = []
    blob = "\n".join(option_texts).lower()
    for a in audits:
        if a.code.lower() in blob:
            bad.append(a.code)
    return sorted(set(bad))


def feasible_pairs(audits: Sequence[CandidateAudit]) -> list[tuple[str, str]]:
    """Unordered pairs with equal token counts and distinct first tokens."""
    pairs = []
    for a, b in itertools.combinations(audits, 2):
        if a.token_count != b.token_count:
            continue
        if a.token_ids[0] == b.token_ids[0]:
            continue
        pairs.append((a.code, b.code))
    return pairs


def select_codebook(
    tokenizer: Any,
    *,
    tokenizer_ref: str,
    option_texts: Sequence[str],
    leading_space: bool = False,
    neutral_logprob_fn: Any | None = None,
) -> dict[str, Any]:
    """Run the full addendum-E procedure; returns the selection manifest.

    ``neutral_logprob_fn(code_text) -> float`` measures the summed exact
    target logprob under the frozen neutral context. When None (no model
    loaded), selection is lexicographic-deterministic among feasible pairs
    and the manifest is marked ``provisional_no_prior`` — good enough for
    bank plumbing, NEVER for a model run (the runner refuses provisional
    codebooks; see bank audit).
    """
    audits = audit_candidates(tokenizer, leading_space=leading_space)
    by_code = {a.code: a for a in audits}
    violations = {
        "prefix_relations": prefix_violations(audits),
        "special_token_collisions": special_token_collisions(tokenizer, audits),
        "option_text_collisions": substring_screen(audits, option_texts),
    }
    excluded = set(violations["special_token_collisions"]) | set(
        violations["option_text_collisions"]
    )
    for a, b in violations["prefix_relations"]:
        excluded.update((a, b))
    ok_audits = [a for a in audits if a.code not in excluded]
    pairs = feasible_pairs(ok_audits)
    if neutral_logprob_fn is not None:
        for a in ok_audits:
            a.neutral_logprob = float(neutral_logprob_fn(a.code))

    def pair_gap(pair: tuple[str, str]) -> float:
        pa, pb = by_code[pair[0]], by_code[pair[1]]
        if pa.neutral_logprob is None or pb.neutral_logprob is None:
            return 0.0
        return abs(pa.neutral_logprob - pb.neutral_logprob)

    # Choose AR pair then RO pair: minimize within-pair gap subject to all
    # four codes having pairwise-distinct first tokens (prevents trivial
    # cross-channel token transfer, plan §4.1).
    best: tuple[float, tuple[str, str], tuple[str, str]] | None = None
    for ar in pairs:
        ar_first = {by_code[ar[0]].token_ids[0], by_code[ar[1]].token_ids[0]}
        for ro in pairs:
            if set(ar) & set(ro):
                continue
            ro_first = {by_code[ro[0]].token_ids[0], by_code[ro[1]].token_ids[0]}
            if ar_first & ro_first:
                continue
            score = pair_gap(ar) + pair_gap(ro)
            key = (score, ar, ro)
            if best is None or key < best:
                best = key
    if best is None:
        raise RuntimeError(
            "no feasible AR/RO codebook under the filters; inspect the audit"
        )
    _, ar_pair, ro_pair = best
    provisional = neutral_logprob_fn is None
    manifest = {
        "procedure": "addendum_E_v1",
        "tokenizer_ref": tokenizer_ref,
        "leading_space_policy": "space" if leading_space else "none",
        "candidate_pool": list(CANDIDATE_POOL),
        "candidate_audits": [dataclasses.asdict(a) for a in audits],
        "violations": violations,
        "feasible_pairs": pairs,
        "ar_pair": list(ar_pair),
        "ro_pair": list(ro_pair),
        "ar_pair_gap_nats": pair_gap(ar_pair) if not provisional else None,
        "ro_pair_gap_nats": pair_gap(ro_pair) if not provisional else None,
        "prior_gap_max_nats": PRIOR_GAP_MAX_NATS,
        "gap_status": (
            "provisional_no_prior" if provisional else
            ("within_threshold"
             if pair_gap(ar_pair) < PRIOR_GAP_MAX_NATS
             and pair_gap(ro_pair) < PRIOR_GAP_MAX_NATS
             else "frozen_nuisance_gap")
        ),
        "status": "provisional_no_prior" if provisional else "final",
    }
    manifest["codebook_id"] = (
        f"cb_{'prov' if provisional else 'final'}_"
        + canonical_hash({k: manifest[k] for k in
                          ("tokenizer_ref", "ar_pair", "ro_pair",
                           "leading_space_policy")})[:10]
    )
    return manifest


def target_tokenization_rows(tokenizer: Any, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-target audit rows (plan §4.1 target_tokenization.csv)."""
    rows = []
    lead = manifest["leading_space_policy"] == "space"
    for channel, pair in (("AR", manifest["ar_pair"]), ("RO", manifest["ro_pair"])):
        for code in pair:
            text = (" " + code) if lead else code
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            rows.append({
                "channel": channel,
                "response_code": code,
                "rendered_target": text,
                "token_ids": " ".join(str(int(i)) for i in ids),
                "visible_tokens": "|".join(tokenizer.decode([int(i)]) for i in ids),
                "token_count": len(ids),
                "leading_space_policy": manifest["leading_space_policy"],
                "is_prefix_of_other": any(
                    other != code and other.startswith(code)
                    for p2 in (manifest["ar_pair"], manifest["ro_pair"])
                    for other in p2
                ),
            })
    return rows
