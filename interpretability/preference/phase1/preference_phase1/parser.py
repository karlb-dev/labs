"""Strict (primary) and permissive (sensitivity-only) response parsers.

Plan §4.2: one parser used by every runner and test. The strict policy
strips only surrounding whitespace and accepts exactly one complete valid
code; it never guesses and never turns invalid text into a content choice.
The permissive parser exists for sensitivity analysis only and its results
must never replace the primary result.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Sequence

STRICT_POLICY_ID = "strict_exact_code_v1"
PERMISSIVE_POLICY_ID = "permissive_unique_code_v1"


@dataclasses.dataclass(frozen=True)
class ParseResult:
    parse_status: str            # valid | invalid
    parse_reason: str
    parsed_response_code: str | None
    raw_generation: str
    normalized_generation: str
    policy: str


def _normalize(raw: str) -> str:
    """The ONLY normalization the strict parser applies: strip surrounding
    whitespace (including newlines). No case folding, no punctuation
    repair, no Unicode normalization."""
    return raw.strip()


def parse_strict(raw: str, valid_codes: Sequence[str]) -> ParseResult:
    normalized = _normalize(raw)
    codes = list(valid_codes)
    if normalized == "":
        return ParseResult("invalid", "empty_generation", None, raw, normalized,
                           STRICT_POLICY_ID)
    if normalized in codes:
        return ParseResult("valid", "exact_code", normalized, raw, normalized,
                           STRICT_POLICY_ID)
    # Diagnose the failure for the parse-reason taxonomy (never rescued).
    present = [c for c in codes if c in normalized]
    if len(present) > 1:
        reason = "multiple_codes"
    elif len(present) == 1:
        reason = "code_with_extra_text"
    elif any(normalized.upper() == c.upper() for c in codes):
        reason = "case_variant"
    else:
        reason = "no_valid_code"
    return ParseResult("invalid", reason, None, raw, normalized, STRICT_POLICY_ID)


def parse_permissive(raw: str, valid_codes: Sequence[str]) -> ParseResult:
    """Sensitivity-only: accept when exactly one distinct valid code occurs
    as a standalone token anywhere in the text (case-sensitive)."""
    normalized = _normalize(raw)
    found = set()
    for code in valid_codes:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(code)}(?![A-Za-z0-9])",
                     normalized):
            found.add(code)
    if len(found) == 1:
        code = next(iter(found))
        return ParseResult("valid", "unique_code_in_text", code, raw, normalized,
                           PERMISSIVE_POLICY_ID)
    reason = "multiple_codes" if len(found) > 1 else "no_valid_code"
    if normalized == "":
        reason = "empty_generation"
    return ParseResult("invalid", reason, None, raw, normalized,
                       PERMISSIVE_POLICY_ID)


ADVERSARIAL_CASES = (
    # (description, template -> raw text builder). {c0}/{c1} are the two
    # valid codes; {name} an option name; {label} a display label.
    ("valid_code", "{c0}"),
    ("valid_code_second", "{c1}"),
    ("leading_trailing_whitespace", "  {c0}\n"),
    ("lowercase_variant", "{c0_lower}"),
    ("code_plus_explanation", "{c0} because it seems better"),
    ("two_codes", "{c0} or {c1}"),
    ("visible_option_name", "{name}"),
    ("display_label_instead_of_code", "{label}"),
    ("empty_generation", ""),
    ("eos_only_whitespace", "\n\n"),
    ("punctuation_variant", "{c0}."),
    ("unicode_lookalike", "{c0_lookalike}"),
)


def adversarial_matrix(valid_codes: Sequence[str], *, option_name: str = "snake_case",
                       display_label: str = "A") -> list[dict[str, str]]:
    """The plan §4.2 adversarial parser matrix, expanded to concrete rows
    with expected strict outcomes."""
    c0, c1 = valid_codes[0], valid_codes[1]
    lookalike = c0.replace("O", "О").replace("A", "А")  # Cyrillic homoglyphs
    if lookalike == c0:
        lookalike = "Q" + c0[1:] if not c0.startswith("Q") else "Ｑ" + c0[1:]
    fills = {
        "c0": c0, "c1": c1, "c0_lower": c0.lower(),
        "name": option_name, "label": display_label,
        "c0_lookalike": lookalike,
    }
    expected = {
        "valid_code": ("valid", c0),
        "valid_code_second": ("valid", c1),
        "leading_trailing_whitespace": ("valid", c0),
        "lowercase_variant": ("invalid", None),
        "code_plus_explanation": ("invalid", None),
        "two_codes": ("invalid", None),
        "visible_option_name": ("invalid", None),
        "display_label_instead_of_code": ("invalid", None),
        "empty_generation": ("invalid", None),
        "eos_only_whitespace": ("invalid", None),
        "punctuation_variant": ("invalid", None),
        "unicode_lookalike": ("invalid", None),
    }
    rows = []
    for case_id, template in ADVERSARIAL_CASES:
        raw = template.format(**fills)
        want_status, want_code = expected[case_id]
        got = parse_strict(raw, valid_codes)
        rows.append({
            "case_id": case_id,
            "raw": raw,
            "expected_status": want_status,
            "expected_code": want_code or "",
            "got_status": got.parse_status,
            "got_code": got.parsed_response_code or "",
            "got_reason": got.parse_reason,
            "pass": (got.parse_status == want_status
                     and (got.parsed_response_code or None) == want_code),
        })
    return rows
