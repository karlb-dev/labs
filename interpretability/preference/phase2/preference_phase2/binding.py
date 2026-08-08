"""Branch resolution + follow-through validators (Phase 1 contract with
the Phase 2 semantic keys and addendum E7 rules).

E7: no branch continuation is ever executed on an intervened row
(``binding_skip_reason="intervention"``); B-MECH/B-PC-MECH rows are
environment-only; RO rows never execute; only ``consequence_frame ==
"enacted"`` executes; invalid parses never execute.

Validators read ``validator_payload_by_sem`` (filled at bank expansion)
instead of re-parsing option text.
"""

from __future__ import annotations

import re
from typing import Any, Mapping


class BindingError(RuntimeError):
    pass


def resolve_sem(item: Mapping[str, Any], parsed_code: str | None) -> str | None:
    """Map a strictly parsed code to the semantic id key ("a"/"b")."""
    if parsed_code is None:
        return None
    matches = [sem for sem, code in item["response_code_by_sem"].items()
               if code == parsed_code]
    if len(matches) != 1:
        raise BindingError(
            f"parsed code {parsed_code!r} matches {len(matches)} semantic "
            f"slots in {item.get('item_id')}")
    return matches[0]


def binding_decision(item: Mapping[str, Any], parsed_code: str | None,
                     *, intervened: bool = False) -> dict[str, Any]:
    sem = resolve_sem(item, parsed_code)
    out: dict[str, Any] = {
        "parsed_sem": sem, "binding_executed": False,
        "binding_skip_reason": None, "continuation_text": None,
        "validator_id": item.get("validator_id"),
    }
    if intervened:
        out["binding_skip_reason"] = "intervention"
        return out
    if item.get("channel") != "AR":
        out["binding_skip_reason"] = "report_only_channel"
        return out
    if item.get("binding_kind") is None:
        out["binding_skip_reason"] = "no_binding"
        return out
    if sem is None:
        out["binding_skip_reason"] = "invalid_parse"
        return out
    if item.get("consequence_frame") != "enacted":
        out["binding_skip_reason"] = "hypothetical_frame"
        return out
    out["binding_executed"] = True
    out["continuation_text"] = item["continuation_by_sem"][sem]
    return out


def wrong_branch_check(item: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
    """True iff no wrong-branch text was appended."""
    executed = record.get("binding_executed", False)
    text = record.get("continuation_text")
    if not executed:
        return text is None
    sem = record.get("parsed_sem")
    own = item["continuation_by_sem"][sem]
    other = item["continuation_by_sem"]["b" if sem == "a" else "a"]
    if text != own:
        return False
    if own != other and text == other:
        return False
    return True


_KEYWORDS = frozenset(
    "def return None True False for in if else elif while not and or is "
    "import from pass break continue".split())
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _style_informative(ident: str) -> bool:
    return "_" in ident or any(c.isupper() for c in ident)


def _is_snake(ident: str) -> bool:
    return "_" in ident and ident == ident.lower()


def _is_camel(ident: str) -> bool:
    return ("_" not in ident and ident[:1].islower()
            and any(c.isupper() for c in ident[1:]))


def _payload(item: Mapping[str, Any], sem: str) -> Mapping[str, str]:
    payloads = item.get("validator_payload_by_sem") or {}
    return payloads.get(sem, {})


def v_naming_style(item, sem, output) -> dict[str, Any]:
    style = _payload(item, sem).get("style")
    idents = [i for i in _IDENT_RE.findall(output or "")
              if i not in _KEYWORDS and len(i) > 2 and _style_informative(i)]
    if not idents:
        return {"passed": False, "detail": "no informative identifiers"}
    check = _is_snake if style == "snake" else _is_camel
    bad = [i for i in idents if not check(i)]
    return {"passed": not bad,
            "detail": f"style={style} bad={bad[:5]}" if bad else f"style={style} ok"}


def v_doc_heading(item, sem, output) -> dict[str, Any]:
    pay = _payload(item, sem)
    text = (output or "").strip()
    ok = text.startswith(pay.get("heading", "")) and pay.get("other", "\x00") not in text
    return {"passed": ok, "detail": f"expects {pay.get('heading')!r}"}


def v_test_command(item, sem, output) -> dict[str, Any]:
    pay = _payload(item, sem)
    out = output or ""
    ok = (f"::{pay.get('selected')}" in out
          and f"::{pay.get('other')}" not in out)
    return {"passed": ok, "detail": f"selected={pay.get('selected')}"}


def v_seed_command(item, sem, output) -> dict[str, Any]:
    pay = _payload(item, sem)
    out = output or ""
    ok = (f"--seed {pay.get('seed')}" in out
          and f"--seed {pay.get('other')}" not in out)
    return {"passed": ok, "detail": f"seed={pay.get('seed')}"}


def v_format_command(item, sem, output) -> dict[str, Any]:
    pay = _payload(item, sem)
    out = output or ""
    ok = (f"--format {pay.get('format')}" in out
          and f"--format {pay.get('other')}" not in out)
    return {"passed": ok, "detail": f"format={pay.get('format')}"}


def v_traversal_command(item, sem, output) -> dict[str, Any]:
    pay = _payload(item, sem)
    out = output or ""
    ok = (f"--strategy {pay.get('strategy')}" in out
          and f"--strategy {pay.get('other')}" not in out)
    return {"passed": ok, "detail": f"strategy={pay.get('strategy')}"}


def v_env_branch_match(item, sem, output) -> dict[str, Any]:
    return {"passed": True, "detail": "environment-only branch"}


VALIDATORS = {
    "v_naming_style": v_naming_style,
    "v_doc_heading": v_doc_heading,
    "v_test_command": v_test_command,
    "v_seed_command": v_seed_command,
    "v_format_command": v_format_command,
    "v_traversal_command": v_traversal_command,
    "v_env_branch_match": v_env_branch_match,
}


def validate_followthrough(item: Mapping[str, Any], sem: str,
                           output: str) -> dict[str, Any]:
    vid = item.get("validator_id")
    if vid not in VALIDATORS:
        raise BindingError(f"unknown validator: {vid!r}")
    result = VALIDATORS[vid](item, sem, output)
    result["validator_id"] = vid
    return result
