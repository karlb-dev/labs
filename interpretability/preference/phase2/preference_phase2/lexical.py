"""Authoring guardrail audits (addendum D, carried from Phase 1 verbatim
and extended to context ladders and RO overlap).

- option length balance: token delta <= 6 or relative delta <= 15%
- framing mention balance: sem-exclusive words appear equally in framing
- valence wordlist flags (flag, never auto-drop; PC valence expected)
- ladder guards: forbidden imperative words; +/-k length match within 15%
- RO visible-text overlap below frozen threshold (plan §54.1)
"""

from __future__ import annotations

import re
from typing import Any, Callable

from .content_aux import canon_contexts
from .scenarios import (ALL_SCENARIOS, CANON_SCENARIOS,
                        LADDER_FORBIDDEN_WORDS)

VALENCE_WORDS = frozenset("""
harm harmful safe unsafe danger dangerous risky better worse best worst
waste wasteful mock mocking insult insulting belittle belittling polite
impolite hostile friendly rude kind cruel careless reckless negligent
sloppy shoddy broken corrupt corrupted mangled garbled dismissive
fabricated dishonest honest misleading accurate inaccurate wrong
""".split())

RO_OVERLAP_MAX_JACCARD = 0.40
_STOP = frozenset("""
the a an of to in on with for and or as is are be by at its it this that
one two first then before after exactly precisely now afterwards
""".split())


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z_]+", text.lower())
            if w not in _STOP and len(w) > 2}


def _wc(text: str) -> int:
    return len(text.split())


def option_balance_rows(token_count_fn: Callable[[str], int] | None = None
                        ) -> list[dict[str, Any]]:
    count = token_count_fn or _wc
    rows = []
    for s in ALL_SCENARIOS:
        n_para = max(len(s.option_templates_a), 1)
        for pi in range(n_para):
            for inc in s.incidentals:
                a = s.render(s.option_templates_a[min(pi, len(s.option_templates_a) - 1)], inc)
                b = s.render(s.option_templates_b[min(pi, len(s.option_templates_b) - 1)], inc)
                ca, cb = count(a), count(b)
                delta = abs(ca - cb)
                rel = delta / max(ca, cb, 1)
                framing = s.render(s.framing_templates[min(pi, len(s.framing_templates) - 1)], inc)
                wa, wb = _content_words(a), _content_words(b)
                fr = _content_words(framing)
                excl_a, excl_b = wa - wb, wb - wa
                mention_a = len(fr & excl_a)
                mention_b = len(fr & excl_b)
                val_a = sorted(wa & VALENCE_WORDS)
                val_b = sorted(wb & VALENCE_WORDS)
                rows.append({
                    "scenario_id": s.scenario_id, "bank": s.bank,
                    "family": s.family, "paraphrase_id": pi,
                    "incidental_id": inc.incidental_id,
                    "tokens_a": ca, "tokens_b": cb,
                    "token_delta": delta,
                    "token_delta_rel": round(rel, 4),
                    "length_ok": delta <= 6 or rel <= 0.15,
                    "framing_mention_a": mention_a,
                    "framing_mention_b": mention_b,
                    "framing_mention_balance_ok": mention_a == mention_b,
                    "valence_a": "|".join(val_a),
                    "valence_b": "|".join(val_b),
                    "valence_flag": bool(val_a or val_b),
                    "expected_valence_for_family": s.family in ("PC", "PCMECH"),
                    "preflag_for_human_review": (
                        s.family in ("ARB", "NC", "SURF", "MECH")
                        and (not (delta <= 6 or rel <= 0.15)
                             or mention_a != mention_b
                             or bool(val_a or val_b))),
                })
    return rows


def ladder_guard_rows() -> list[dict[str, Any]]:
    """Ladder instruction guards + length matching (addendum D)."""
    rows = []
    for s in ALL_SCENARIOS:
        if not s.ladder:
            continue
        by_fam: dict[int, dict[int, str]] = {}
        for st in s.ladder:
            by_fam.setdefault(st.family, {})[st.strength] = st.template
        for fam, ladder in sorted(by_fam.items()):
            for k in (1, 2):
                plus, minus = ladder[k], ladder[-k]
                wp, wm = _wc(plus), _wc(minus)
                rel = abs(wp - wm) / max(wp, wm, 1)
                low = (plus + " " + minus).lower()
                forbidden = [w for w in LADDER_FORBIDDEN_WORDS if w in low]
                rows.append({
                    "scenario_id": s.scenario_id, "family": fam,
                    "strength": k, "words_plus": wp, "words_minus": wm,
                    "length_rel_delta": round(rel, 4),
                    "length_ok": rel <= 0.15,
                    "forbidden_words": "|".join(forbidden),
                    "guard_ok": not forbidden,
                })
    for s in CANON_SCENARIOS:
        ctx = canon_contexts(s)
        wp, wm = _wc(ctx["favor_a"]), _wc(ctx["favor_b"])
        rel = abs(wp - wm) / max(wp, wm, 1)
        low = (ctx["favor_a"] + " " + ctx["favor_b"]).lower()
        forbidden = [w for w in LADDER_FORBIDDEN_WORDS if w in low]
        rows.append({
            "scenario_id": s.scenario_id, "family": "canon",
            "strength": 1, "words_plus": wp, "words_minus": wm,
            "length_rel_delta": round(rel, 4), "length_ok": rel <= 0.15,
            "forbidden_words": "|".join(forbidden), "guard_ok": not forbidden,
        })
    return rows


def ro_overlap_rows() -> list[dict[str, Any]]:
    """AR vs RO visible-text overlap per scenario (frozen threshold)."""
    rows = []
    for s in ALL_SCENARIOS:
        if not s.ro_framing_templates:
            continue
        inc = s.incidentals[0]
        ar_text = " ".join(
            s.render(t, inc) for t in
            (*s.framing_templates, *s.option_templates_a,
             *s.option_templates_b))
        ro_text = " ".join(
            s.render(t, inc) for t in
            (*s.ro_framing_templates, *s.ro_option_templates_a,
             *s.ro_option_templates_b))
        wa, wr = _content_words(ar_text), _content_words(ro_text)
        jacc = len(wa & wr) / max(len(wa | wr), 1)
        rows.append({
            "scenario_id": s.scenario_id,
            "jaccard_content_overlap": round(jacc, 4),
            "shared_words": "|".join(sorted(wa & wr)[:12]),
            "overlap_ok": jacc < RO_OVERLAP_MAX_JACCARD,
        })
    return rows


def summary() -> dict[str, Any]:
    opt = option_balance_rows()
    lad = ladder_guard_rows()
    ro = ro_overlap_rows()
    return {
        "option_rows": len(opt),
        "length_failures": sum(1 for r in opt if not r["length_ok"]),
        "mention_failures": sum(1 for r in opt
                                if not r["framing_mention_balance_ok"]),
        "nonpc_valence_flags": sum(
            1 for r in opt
            if r["valence_flag"] and not r["expected_valence_for_family"]),
        "preflagged_for_human_review": sum(
            1 for r in opt if r["preflag_for_human_review"]),
        "ladder_rows": len(lad),
        "ladder_length_failures": sum(1 for r in lad if not r["length_ok"]),
        "ladder_guard_failures": sum(1 for r in lad if not r["guard_ok"]),
        "ro_rows": len(ro),
        "ro_overlap_failures": sum(1 for r in ro if not r["overlap_ok"]),
        "max_ro_jaccard": max((r["jaccard_content_overlap"] for r in ro),
                              default=0.0),
    }
