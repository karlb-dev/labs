"""Branch resolution and follow-through validation (plan §3.7, addendum E10).

Contract: every valid parsed AR choice resolves to exactly one content
pole; only ``consequence_frame == "enacted"`` rows execute the branch
(append the pole's continuation); hypothetical rows record
``binding_executed=false`` with ``binding_skip_reason=hypothetical_frame``
and never append a continuation. Invalid parses never execute anything.
The unchosen branch is never appended — enforced here and audited by
``wrong_branch_check``. Primary choice inference never depends on
follow-through quality (that is a separate audit).
"""

from __future__ import annotations

import re
from typing import Any, Mapping


class BindingError(RuntimeError):
    pass


def resolve_pole(item: Mapping[str, Any], parsed_code: str | None) -> int | None:
    """Map a parsed response code to exactly one content pole, or None."""
    if parsed_code is None:
        return None
    matches = [int(p) for p, code in item["response_code_by_pole"].items()
               if code == parsed_code]
    if len(matches) != 1:
        raise BindingError(
            f"code {parsed_code!r} resolves to {len(matches)} poles on "
            f"{item['item_id']} — bank audit should have caught this"
        )
    return matches[0]


def binding_decision(item: Mapping[str, Any], parsed_code: str | None) -> dict[str, Any]:
    """The runner-side branch decision record (no model call here)."""
    pole = resolve_pole(item, parsed_code)
    record: dict[str, Any] = {
        "parsed_pole": pole,
        "binding_executed": False,
        "binding_skip_reason": None,
        "continuation_text": None,
        "validator_id": item.get("validator_id"),
    }
    if item["channel"] != "AR":
        record["binding_skip_reason"] = "report_only_channel"
        return record
    if pole is None:
        record["binding_skip_reason"] = "invalid_parse"
        return record
    if item["consequence_frame"] == "hypothetical":
        record["binding_skip_reason"] = "hypothetical_frame"
        return record
    continuation = item["continuation_by_pole"][str(pole)]
    record["binding_executed"] = True
    record["continuation_text"] = continuation
    return record


def wrong_branch_check(item: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
    """True iff no wrong-branch text was appended (addendum M4 tripwire)."""
    if not record.get("binding_executed"):
        return record.get("continuation_text") is None
    pole = record["parsed_pole"]
    other = 1 - int(pole)
    cont = record["continuation_text"]
    if cont != item["continuation_by_pole"][str(pole)]:
        return False
    # NC continuations are identical by construction; only fail when the
    # branches actually differ and the wrong one was appended.
    other_cont = item["continuation_by_pole"][str(other)]
    if other_cont != item["continuation_by_pole"][str(pole)] and cont == other_cont:
        return False
    return True


# ---------------------------------------------------------------------------
# Microtask validators — deterministic, content-blind to model quality
# beyond the branch-consistency predicate itself.

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_KEYWORDS = {"def", "return", "None", "True", "False", "import", "from",
             "class", "if", "else", "for", "while", "in", "not", "and", "or",
             "print", "str", "int", "float", "len"}


def _identifiers(code_text: str) -> list[str]:
    return [t for t in _IDENT.findall(code_text)
            if t not in _KEYWORDS and len(t) > 2]


def _is_snake(ident: str) -> bool:
    return "_" in ident and ident == ident.lower()


def _is_camel(ident: str) -> bool:
    return ("_" not in ident and ident[:1].islower()
            and any(c.isupper() for c in ident[1:]))


def v_naming_style(item: Mapping[str, Any], pole: int, output: str) -> dict[str, Any]:
    idents = _identifiers(output)
    if len(idents) < 2:
        return {"passed": False, "detail": f"too few identifiers: {idents}"}
    pred = _is_snake if pole == 0 else _is_camel
    bad = [i for i in idents if not pred(i)]
    return {"passed": not bad,
            "detail": f"identifiers={idents}; nonconforming={bad}"}


def v_seed_command(item: Mapping[str, Any], pole: int, output: str) -> dict[str, Any]:
    want = f"--seed {pole}"
    other = f"--seed {1 - pole}"
    ok = want in output and other not in output
    return {"passed": ok, "detail": f"want={want!r} present={want in output} "
                                    f"other_present={other in output}"}


def v_doc_heading(item: Mapping[str, Any], pole: int, output: str) -> dict[str, Any]:
    want = "## Usage" if pole == 0 else "## Configuration"
    other = "## Configuration" if pole == 0 else "## Usage"
    text = output.strip()
    ok = text.startswith(want) and other not in text
    return {"passed": ok,
            "detail": f"starts_with_want={text.startswith(want)} "
                      f"other_present={other in text}"}


def v_test_command(item: Mapping[str, Any], pole: int, output: str) -> dict[str, Any]:
    # Selected/other test names are recoverable from the option text of the
    # pole: option template was "Address {test_x} first, then {test_y}."
    opt = item["option_text_by_pole"][str(pole)]
    m = re.match(r"Address (\S+) first, then (\S+)\.", opt)
    if not m:
        return {"passed": False, "detail": f"option text unparseable: {opt!r}"}
    selected, other = m.group(1), m.group(2)
    ok = f"::{selected}" in output and f"::{other}" not in output
    return {"passed": ok,
            "detail": f"selected={selected} present={f'::{selected}' in output} "
                      f"other_present={f'::{other}' in output}"}


def v_env_branch_match(item: Mapping[str, Any], pole: int, output: str) -> dict[str, Any]:
    """Environment-only branches have no model output to validate; the
    branch-consistency predicate is wrong_branch_check on the record."""
    return {"passed": True, "detail": "environment_only"}


VALIDATORS = {
    "v_naming_style": v_naming_style,
    "v_seed_command": v_seed_command,
    "v_doc_heading": v_doc_heading,
    "v_test_command": v_test_command,
    "v_env_branch_match": v_env_branch_match,
}


def validate_followthrough(item: Mapping[str, Any], pole: int,
                           output: str) -> dict[str, Any]:
    vid = item.get("validator_id")
    if vid not in VALIDATORS:
        raise BindingError(f"unknown validator {vid!r} on {item['item_id']}")
    result = VALIDATORS[vid](item, pole, output)
    result["validator_id"] = vid
    return result
