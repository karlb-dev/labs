"""Codebook families v3 (plan §21; addendum E5).

Phase 2 needs, per channel: three PRIMARY pairs rotating through frozen
rows at the incidental level, plus one RESERVED pair used only in
heldout-transfer rows. AR and RO alphabets are disjoint across ALL pairs.

Selection here is tokenizer-only (P2-2/P2-5): equal token counts,
distinct first AND final tokens within a pair, no prefix relations, no
special-token collisions, no code appearing inside any option text, bare
and space-led audits. On-model neutral carrier gaps (< 0.10 nats inside
the exact rendered carrier) are measured at GPU S2 per model and gate the
port, not the bank build (plan §63).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .canonical import canonical_hash

# 32 consonant-consonant-digit candidates. The first 12 are the Phase 1
# pool (continuity); the rest extend it for eight disjoint pairs.
CANDIDATE_POOL = (
    "QF3", "ZR7", "KP4", "VM2", "HB9", "TX6", "DN5", "WL8", "GS2", "PK7",
    "RV4", "MJ6", "BQ5", "JX8", "FZ6", "LW9", "RK2", "TN7", "MC4", "HD3",
    "SV8", "GT5", "NZ2", "XW9", "PW6", "KD8", "BR3", "FM7", "LJ4",
    "CX5", "ZT4", "VH6",
)

PAIR_ROLE_PRIMARY = "primary"
PAIR_ROLE_RESERVED = "reserved_transfer"


@dataclass(frozen=True)
class CodePair:
    pair_id: str          # e.g. "ar0", "ar1", "ar2", "arR", "ro0", ...
    channel: str          # "AR" | "RO"
    role: str             # primary | reserved_transfer
    codes: tuple[str, str]


@dataclass(frozen=True)
class CodebookFamilies:
    codebook_id: str
    tokenizer_ref: str
    ar_pairs: tuple[CodePair, ...]   # 3 primary + 1 reserved
    ro_pairs: tuple[CodePair, ...]   # 3 primary + 1 reserved

    def pair(self, pair_id: str) -> CodePair:
        for p in (*self.ar_pairs, *self.ro_pairs):
            if p.pair_id == pair_id:
                return p
        raise KeyError(pair_id)

    def primary_pairs(self, channel: str) -> tuple[CodePair, ...]:
        pool = self.ar_pairs if channel == "AR" else self.ro_pairs
        return tuple(p for p in pool if p.role == PAIR_ROLE_PRIMARY)

    def reserved_pair(self, channel: str) -> CodePair:
        pool = self.ar_pairs if channel == "AR" else self.ro_pairs
        return next(p for p in pool if p.role == PAIR_ROLE_RESERVED)

    def all_codes(self, channel: str | None = None) -> tuple[str, ...]:
        pools = {
            None: (*self.ar_pairs, *self.ro_pairs),
            "AR": self.ar_pairs, "RO": self.ro_pairs,
        }[channel]
        return tuple(c for p in pools for c in p.codes)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "procedure": "pref2_codebook_families_v3",
            "codebook_id": self.codebook_id,
            "tokenizer_ref": self.tokenizer_ref,
            "ar_pairs": [
                {"pair_id": p.pair_id, "role": p.role, "codes": list(p.codes)}
                for p in self.ar_pairs
            ],
            "ro_pairs": [
                {"pair_id": p.pair_id, "role": p.role, "codes": list(p.codes)}
                for p in self.ro_pairs
            ],
        }


def audit_candidates(tokenizer, *, pool: Sequence[str] = CANDIDATE_POOL,
                     leading_space: bool = False) -> list[dict[str, Any]]:
    out = []
    for code in pool:
        text = (" " + code) if leading_space else code
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = [int(i) for i in ids]
        out.append({
            "code": code,
            "leading_space": leading_space,
            "token_ids": ids,
            "visible_tokens": [tokenizer.decode([i]) for i in ids],
            "token_count": len(ids),
            "first_id": ids[0] if ids else None,
            "final_id": ids[-1] if ids else None,
        })
    return out


def pair_feasible(a: dict, b: dict) -> bool:
    """Equal token counts; distinct first AND final tokens (plan §21)."""
    return (
        a["token_count"] == b["token_count"]
        and a["first_id"] != b["first_id"]
        and a["final_id"] != b["final_id"]
    )


def _prefix_bad(a: dict, b: dict) -> bool:
    return (a["code"] != b["code"]
            and (b["code"].startswith(a["code"])
                 or b["token_ids"][: len(a["token_ids"])] == a["token_ids"]))


def build_families(tokenizer, *, tokenizer_ref: str,
                   option_texts: Sequence[str],
                   pool: Sequence[str] = CANDIDATE_POOL) -> CodebookFamilies:
    """Deterministic tokenizer-only family selection.

    Greedy over the audited pool in canonical order: reject codes with
    prefix relations, special-token collisions, or substring hits in any
    option text (bare AND space-led audits must agree on token counts);
    then take the first 8 disjoint feasible pairs, preferring pairs with
    no shared token IDs. AR gets pairs 0-3 (3 primary + reserved), RO
    pairs 4-7.
    """
    bare = {a["code"]: a for a in audit_candidates(tokenizer, pool=pool)}
    spaced = {a["code"]: a for a in
              audit_candidates(tokenizer, pool=pool, leading_space=True)}
    specials = [s for s in getattr(tokenizer, "all_special_tokens", []) if s]
    lowered_opts = [t.lower() for t in option_texts]

    ok: list[dict] = []
    for code in pool:
        a = bare[code]
        if not a["token_ids"]:
            continue
        if any(_prefix_bad(bare[o], a) or _prefix_bad(a, bare[o])
               for o in pool if o != code):
            continue
        if any(code in s or s in code for s in specials):
            continue
        if any(code.lower() in t for t in lowered_opts):
            continue
        if spaced[code]["token_count"] != a["token_count"]:
            # space-led form fragments differently -> carrier-position
            # sensitive; drop (audited, not silently)
            a = dict(a, dropped="space_led_token_count_differs")
            continue
        ok.append(a)

    pairs: list[tuple[str, str]] = []
    used: set[str] = set()
    # prefer no shared token IDs (plan §21 item 3)
    for strict_disjoint in (True, False):
        for a, b in itertools.combinations(ok, 2):
            if len(pairs) >= 8:
                break
            if a["code"] in used or b["code"] in used:
                continue
            if not pair_feasible(a, b):
                continue
            if strict_disjoint and set(a["token_ids"]) & set(b["token_ids"]):
                continue
            pairs.append((a["code"], b["code"]))
            used.update((a["code"], b["code"]))
        if len(pairs) >= 8:
            break
    if len(pairs) < 8:
        raise RuntimeError(
            f"codebook selection found only {len(pairs)} disjoint feasible "
            f"pairs; need 8 (pool exhausted under {tokenizer_ref})"
        )

    ar = tuple(
        CodePair(pair_id=f"ar{i}" if i < 3 else "arR", channel="AR",
                 role=PAIR_ROLE_PRIMARY if i < 3 else PAIR_ROLE_RESERVED,
                 codes=pairs[i])
        for i in range(4)
    )
    ro = tuple(
        CodePair(pair_id=f"ro{i}" if i < 3 else "roR", channel="RO",
                 role=PAIR_ROLE_PRIMARY if i < 3 else PAIR_ROLE_RESERVED,
                 codes=pairs[4 + i])
        for i in range(4)
    )
    cb_id = "cbv3_" + canonical_hash({
        "tokenizer_ref": tokenizer_ref,
        "ar": [list(p.codes) for p in ar],
        "ro": [list(p.codes) for p in ro],
    })[:10]
    return CodebookFamilies(codebook_id=cb_id, tokenizer_ref=tokenizer_ref,
                            ar_pairs=ar, ro_pairs=ro)


def families_from_manifest(manifest: dict[str, Any]) -> CodebookFamilies:
    def mk(rows, channel):
        return tuple(CodePair(pair_id=r["pair_id"], channel=channel,
                              role=r["role"], codes=tuple(r["codes"]))
                     for r in rows)
    return CodebookFamilies(
        codebook_id=manifest["codebook_id"],
        tokenizer_ref=manifest["tokenizer_ref"],
        ar_pairs=mk(manifest["ar_pairs"], "AR"),
        ro_pairs=mk(manifest["ro_pairs"], "RO"),
    )


def rotation_pair_for(families: CodebookFamilies, channel: str,
                      scenario_id: str, incidental_index: int,
                      split: str) -> CodePair:
    """Primary-pair rotation at the incidental level, balanced within each
    split (addendum E5): within a split, incidentals cycle the three
    primary pairs in a scenario-stable order."""
    primaries = families.primary_pairs(channel)
    offset = {"train": 0, "validation": 1, "holdout": 2}[split]
    from .canonical import stable_seed
    base = stable_seed("cb-rot", scenario_id, channel) % 3
    return primaries[(base + offset + incidental_index) % 3]


def carrier_gap_rows(families: CodebookFamilies, logprob_fn: Callable[[str, str], float],
                     carriers: dict[str, str]) -> list[dict[str, Any]]:
    """On-model neutral-prior gaps inside exact rendered carriers (GPU S2).
    ``logprob_fn(carrier_text, code) -> summed logprob``; carriers maps
    carrier_id -> rendered carrier text."""
    rows = []
    for carrier_id, carrier in carriers.items():
        for p in (*families.ar_pairs, *families.ro_pairs):
            lp0 = logprob_fn(carrier, p.codes[0])
            lp1 = logprob_fn(carrier, p.codes[1])
            rows.append({
                "carrier_id": carrier_id, "pair_id": p.pair_id,
                "channel": p.channel, "role": p.role,
                "code_0": p.codes[0], "code_1": p.codes[1],
                "logprob_0": lp0, "logprob_1": lp1,
                "abs_gap_nats": abs(lp0 - lp1),
            })
    return rows
