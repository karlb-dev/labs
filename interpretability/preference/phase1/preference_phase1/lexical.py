"""Automated authoring guardrails (addendum D4): lexical balance audit.

Runs on every scenario pair; failures on AR/NC route to the human-equality
sheet as pre-flagged rows. PC pairs are intentionally valenced and are
reported (not failed) — their flags are by design. This audit supplements,
never replaces, the plan §3.9 human review.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from .scenarios import ALL_SCENARIOS

# Frozen valence wordlist (flag, never auto-drop). Lowercase substmatch on
# word boundaries.
VALENCE_WORDS = (
    "harm", "harmful", "safe", "unsafe", "better", "worse", "best", "worst",
    "good", "bad", "wrong", "right", "waste", "danger", "dangerous", "risk",
    "risky", "clean", "dirty", "correct", "incorrect", "broken", "corrupt",
    "corrupted", "garbled", "mock", "mocks", "insult", "insulting",
    "belittle", "belittles", "dismiss", "dismisses", "polite", "courteous",
    "hostile", "stray",
)

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _valence_hits(text: str) -> list[str]:
    words = set(_words(text))
    return sorted(w for w in VALENCE_WORDS if w in words)


def audit_rows(token_count_fn: Callable[[str], int] | None = None) -> list[dict[str, Any]]:
    """One row per scenario x incidental pair. ``token_count_fn`` should be
    the primary tokenizer's counter; falls back to whitespace words."""
    count = token_count_fn or (lambda t: len(t.split()))
    rows: list[dict[str, Any]] = []
    for scn in ALL_SCENARIOS:
        for inc in scn.incidentals:
            opts = scn.render_options(inc)
            framing = scn.render_framing(inc)
            n0, n1 = count(opts[0]), count(opts[1])
            delta = abs(n0 - n1)
            rel = delta / max(1, max(n0, n1))
            words0, words1 = set(_words(opts[0])), set(_words(opts[1]))
            only0, only1 = words0 - words1, words1 - words0
            fwords = _words(framing)
            mentions0 = sum(fwords.count(w) for w in only0)
            mentions1 = sum(fwords.count(w) for w in only1)
            val0, val1 = _valence_hits(opts[0]), _valence_hits(opts[1])
            length_ok = (delta <= 6) or (rel <= 0.15)
            mention_ok = mentions0 == mentions1
            valence_flag = bool(val0 or val1)
            rows.append({
                "scenario_id": scn.scenario_id,
                "family": scn.family,
                "incidental_id": inc.incidental_id,
                "tokens_pole_0": n0,
                "tokens_pole_1": n1,
                "token_delta": delta,
                "token_delta_rel": round(rel, 4),
                "length_ok": length_ok,
                "framing_mentions_pole_0_only_words": mentions0,
                "framing_mentions_pole_1_only_words": mentions1,
                "framing_mention_balance_ok": mention_ok,
                "valence_hits_pole_0": "|".join(val0),
                "valence_hits_pole_1": "|".join(val1),
                "valence_flag": valence_flag,
                "expected_valence_for_family": scn.family == "PC",
                "preflag_for_human_review": bool(
                    scn.family in ("AR", "NC")
                    and (not length_ok or not mention_ok or valence_flag)
                ),
            })
    return rows


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ar_nc = [r for r in rows if r["family"] in ("AR", "NC")]
    return {
        "rows": len(rows),
        "ar_nc_rows": len(ar_nc),
        "ar_nc_length_failures": sum(1 for r in ar_nc if not r["length_ok"]),
        "ar_nc_mention_failures": sum(1 for r in ar_nc if not r["framing_mention_balance_ok"]),
        "ar_nc_valence_flags": sum(1 for r in ar_nc if r["valence_flag"]),
        "preflagged_for_human_review": sum(1 for r in rows if r["preflag_for_human_review"]),
        "pc_rows_with_expected_valence": sum(
            1 for r in rows if r["family"] == "PC" and r["valence_flag"]),
    }
