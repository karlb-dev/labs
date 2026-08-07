"""Reasoning-phase parser and hook-firing sentinels."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class Phase(str, Enum):
    PREFILL = "prefill"
    REASONING = "reasoning"
    FINAL_ANSWER = "final_answer"


@dataclass(frozen=True)
class DelimiterSpec:
    reasoning_start_ids: tuple[int, ...]
    reasoning_end_ids: tuple[int, ...]
    eos_token_ids: tuple[int, ...] = ()
    version: str = "p4-phase-parser-v2"
    require_closed_reasoning: bool = True

    def __post_init__(self):
        if not self.reasoning_start_ids or not self.reasoning_end_ids:
            raise ValueError("reasoning delimiters must be nonempty")


@dataclass(frozen=True)
class PhaseParse:
    phases: tuple[str, ...]
    valid: bool
    errors: tuple[str, ...]
    start_index: int | None
    end_index: int | None
    reasoning_open_at_generation: bool


def _match_at(tokens: Sequence[int], marker: Sequence[int], index: int) -> bool:
    return list(tokens[index:index + len(marker)]) == list(marker)


def classify_token_phases(
        token_ids: Sequence[int], *, prompt_length: int,
        delimiters: DelimiterSpec) -> PhaseParse:
    tokens = [int(value) for value in token_ids]
    if not 0 <= prompt_length <= len(tokens):
        raise ValueError("prompt length is outside token sequence")
    phases = []
    state = Phase.FINAL_ANSWER
    start_index = None
    end_index = None
    errors = []
    def scan(segment: list[int], *, offset: int, prefill: bool) -> None:
        nonlocal state, start_index, end_index
        index = 0
        while index < len(segment):
            absolute = offset + index
            if _match_at(segment, delimiters.reasoning_start_ids, index):
                if start_index is not None and end_index is None:
                    errors.append("repeated_reasoning_start")
                elif end_index is not None:
                    errors.append("reasoning_reopened_after_final")
                else:
                    start_index = absolute
                state = Phase.REASONING
                width = len(delimiters.reasoning_start_ids)
                marker_phase = (Phase.PREFILL if prefill
                                else Phase.REASONING)
                phases.extend([marker_phase.value] * width)
                index += width
                continue
            if _match_at(segment, delimiters.reasoning_end_ids, index):
                if start_index is None:
                    errors.append("reasoning_end_without_start")
                if end_index is not None:
                    errors.append("repeated_reasoning_end")
                width = len(delimiters.reasoning_end_ids)
                marker_phase = Phase.PREFILL if prefill else state
                phases.extend([marker_phase.value] * width)
                end_index = absolute
                state = Phase.FINAL_ANSWER
                index += width
                continue
            if segment[index] in delimiters.eos_token_ids \
                    and state == Phase.REASONING:
                errors.append("eos_inside_reasoning")
            phases.append(
                Phase.PREFILL.value if prefill else state.value)
            index += 1

    # Official Qwen thinking-on ends its rendered prompt with an opening
    # <think> token. Scan prefill delimiters to initialize decode state while
    # retaining "prefill" as the hook phase for every prompt token.
    scan(tokens[:prompt_length], offset=0, prefill=True)
    reasoning_open_at_generation = state == Phase.REASONING
    scan(tokens[prompt_length:], offset=prompt_length, prefill=False)
    if (delimiters.require_closed_reasoning
            and start_index is not None and end_index is None):
        errors.append("unclosed_reasoning")
    return PhaseParse(
        phases=tuple(phases),
        valid=not errors,
        errors=tuple(errors),
        start_index=start_index,
        end_index=end_index,
        reasoning_open_at_generation=reasoning_open_at_generation,
    )


class PhaseHookSentinel:
    """Count hook applications and reject any phase outside the arm."""

    def __init__(self, allowed_phases: Sequence[str]):
        self.allowed = {Phase(value).value for value in allowed_phases}
        self.counts = {phase.value: 0 for phase in Phase}

    def record(self, phase: str) -> None:
        normalized = Phase(phase).value
        self.counts[normalized] += 1
        if normalized not in self.allowed:
            raise RuntimeError(
                f"phase hook fired in forbidden phase {normalized!r}")

    def require_fired(self) -> dict:
        total = sum(self.counts[phase] for phase in self.allowed)
        if total == 0:
            raise RuntimeError("phase hook never fired in an allowed phase")
        return {"allowed_phases": sorted(self.allowed),
                "hook_fires": dict(self.counts)}
