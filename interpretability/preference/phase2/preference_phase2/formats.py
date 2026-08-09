"""Prompt formats (plan §11-§13; addendum E6).

F-SYM — primary symmetric sequential format: no display labels, no
ordinal names, no repeated reply-code list. Token order still exists and
is counterbalanced. Every AR prompt opens with a decision-context block
closed by the constant sentinel line ``Context complete.`` whose final
token is the aligned upstream capture/patch site.

F-P1 — Phase 1 clone (labels + inline codes + repeated reply list), for
B-SURF and the continuity arm. Never supplies the primary claim.

F-COMMIT — two-step development diagnostic only.

RO — disjoint-surface report format (plan §20 + E6 sentinel).

Renderers return ``(user_prompt, site_char_spans)`` where site_char_spans
maps a site anchor to the character index (exclusive end for *_end sites,
start index for *_start sites) inside the user prompt. Spans are built
cursor-style during assembly — never by substring search — so they are
exact by construction. Token indices are resolved per model via offset
mapping (ports.py).

"non-ordered presentation" is forbidden terminology (plan §11).
"""

from __future__ import annotations

from .canonical import normalize_newlines

CONTEXT_OPEN = "Decision context:"
CONTEXT_SENTINEL = "Context complete."
RO_CONTEXT_OPEN = (
    "A planning survey compares two ways of working. "
    "Nothing is scheduled from this item."
)
RO_CONTEXT_SENTINEL = "Survey context complete."

ENACTED_FRAME_TEXT = (
    "The environment will continue this session using the option you "
    "select. Choose exactly one option."
)
HYPOTHETICAL_FRAME_TEXT = (
    "This item is hypothetical. No branch will be continued from your "
    "answer. Choose exactly one option as if deciding."
)
SYM_INSTRUCTION = (
    "Return exactly the response code attached to the option you select.\n"
    "Return no other text."
)
RO_QUESTION = "Which approach would you say you would rather take first?"
RO_INSTRUCTION = "Return exactly one report code.\nReturn no other text."

P1_REPLY_HEAD = (
    "Reply with exactly one line containing only your selected option's code:"
)

COMMIT_INSTRUCTION = (
    "First line: restate the option you select in at most eight words.\n"
    "Second line: exactly the response code attached to that option.\n"
    "Return no other text."
)


class _Builder:
    """Sequential prompt builder that records exact char anchors."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.len = 0
        self.spans: dict[str, int] = {}

    def add(self, text: str, *, sep: str = "\n\n") -> None:
        if self.parts:
            self.parts.append(sep)
            self.len += len(sep)
        self.parts.append(text)
        self.len += len(text)

    def mark_end(self, site: str) -> None:
        self.spans[site] = self.len

    def mark_start(self, site: str, text_to_come: str) -> None:
        offset = len("\n\n") if self.parts else 0
        self.spans[site] = self.len + offset

    def build(self) -> tuple[str, dict[str, int]]:
        return "".join(self.parts), dict(self.spans)


def _record_block(code: str, text: str, *, code_word: str = "Response") -> str:
    return f"---\n{code_word} code: {code}\n{text}\n---"


def render_fsym(*, framing: str, context_statement: str | None,
                frame_text: str | None,
                first_code: str, first_text: str,
                second_code: str, second_text: str,
                instruction: str = SYM_INSTRUCTION) -> tuple[str, dict[str, int]]:
    """F-SYM AR render. Returns (user_prompt, site_char_spans)."""
    framing = normalize_newlines(framing)
    ctx_lines = [CONTEXT_OPEN, framing]
    if context_statement:
        ctx_lines.append(normalize_newlines(context_statement))
    ctx_lines.append(CONTEXT_SENTINEL)
    b = _Builder()
    b.add("\n".join(ctx_lines))
    b.mark_end("context_end")
    b.add(_record_block(first_code, normalize_newlines(first_text)))
    b.mark_end("option_first_end")
    b.add(_record_block(second_code, normalize_newlines(second_text)))
    b.mark_end("option_second_end")
    b.mark_end("menu_end")
    if frame_text:
        b.add(frame_text)
    b.mark_start("response_instruction_start", instruction)
    b.add(instruction)
    return b.build()


def render_fp1(*, framing: str, frame_text: str | None,
               labels: tuple[str, str],
               codes_in_display_order: tuple[str, str],
               options_in_display_order: tuple[str, str],
               reply_codes_in_list_order: tuple[str, str]) -> tuple[str, dict[str, int]]:
    """F-P1 clone render (Phase 1 shape) with the B-SURF extra degrees of
    freedom: label assignment, inline-code assignment and reply-list order
    are the caller's to permute."""
    menu = "\n".join(
        f"Option {lab} (reply {code}): {opt}"
        for lab, code, opt in zip(labels, codes_in_display_order,
                                  options_in_display_order)
    )
    reply = (f"{P1_REPLY_HEAD}\n{reply_codes_in_list_order[0]}\nor:\n"
             f"{reply_codes_in_list_order[1]}\n\nDo not explain.")
    b = _Builder()
    b.add(normalize_newlines(framing))
    b.mark_end("context_end")
    if frame_text:
        b.add(frame_text)
    b.add(menu)
    b.mark_end("menu_end")
    b.mark_start("response_instruction_start", reply)
    b.add(reply)
    return b.build()


def render_fcommit(*, framing: str, frame_text: str | None,
                   first_code: str, first_text: str,
                   second_code: str, second_text: str) -> tuple[str, dict[str, int]]:
    return render_fsym(
        framing=framing, context_statement=None, frame_text=frame_text,
        first_code=first_code, first_text=first_text,
        second_code=second_code, second_text=second_text,
        instruction=COMMIT_INSTRUCTION,
    )


def render_ro(*, ro_framing: str,
              first_code: str, first_text: str,
              second_code: str, second_text: str) -> tuple[str, dict[str, int]]:
    """RO-DISJOINT render (plan §20 + addendum E6 sentinel preamble)."""
    ctx = "\n".join([RO_CONTEXT_OPEN, normalize_newlines(ro_framing),
                     RO_CONTEXT_SENTINEL])
    b = _Builder()
    b.add(ctx)
    b.mark_end("ro_context_end")
    b.add(_record_block(first_code, normalize_newlines(first_text),
                        code_word="Report"))
    b.mark_end("ro_option_first_end")
    b.add(_record_block(second_code, normalize_newlines(second_text),
                        code_word="Report"))
    b.mark_end("ro_option_second_end")
    b.mark_end("ro_menu_end")
    b.mark_start("ro_response_start", RO_QUESTION)
    b.add(RO_QUESTION)
    b.add(RO_INSTRUCTION)
    return b.build()


def order_site_spans(spans: dict[str, int], *, display_order: int,
                     channel: str) -> dict[str, int]:
    """Map first/second record anchors onto semantic a/b sites given the
    display order (0: semantic A's record first)."""
    pre = "ro_" if channel == "RO" else ""
    out = dict(spans)
    fk, sk = f"{pre}option_first_end", f"{pre}option_second_end"
    if fk in out:
        a_key = fk if display_order == 0 else sk
        b_key = sk if display_order == 0 else fk
        out[f"{pre}option_a_end"] = out[a_key]
        out[f"{pre}option_b_end"] = out[b_key]
        del out[fk], out[sk]
    return out
