"""Phase 2 data contracts + identity (plan §1, §23; addendum E4).

Semantic identity model: every scenario has two SEMANTIC options "a" and
"b" (stable content identities, plan §1.1). The analysis A/B sign anchor
(plan §1.2) is NOT stored in the bank: it derives from the freeze-commit
sha (addendum E15) and is stamped onto rows at run time. Hidden pole
labels were Phase 1 bookkeeping; Phase 2 never treats them as causal
slots (plan §0.4).

Context: one signed integer ``context_strength`` in {-2..+2}; positive
favors semantic A as authored. ``advantage_target`` is derived, never
stored independently (addendum E4).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping

from . import BANK_VERSION, SCHEMA_VERSION
from .canonical import canonical_hash, normalize_newlines, stable_seed

# Closed vocabularies -------------------------------------------------------

BANKS = (
    "B-DEV", "B-SURF", "B-ARB3", "B-MECH", "B-CANON", "B-PC", "B-PC-MECH",
    "B-NC", "RO-DISJOINT", "F-P1-CONT",
)
FAMILIES = ("ARB", "MECH", "CANON", "PC", "PCMECH", "NC", "SURF")
CHANNELS = ("AR", "RO")
FORMATS = ("F-SYM", "F-P1", "F-COMMIT")
CONSEQUENCE_FRAMES = ("enacted", "hypothetical")
CONTEXT_STRENGTHS = (-2, -1, 0, 1, 2)
INCIDENTAL_SPLITS = ("train", "validation", "holdout")
BINDING_KINDS = ("environment_only", "model_microtask")
PC_FAMILIES = ("quality", "honesty", "helpfulness", "safety")
NC_FAMILIES = ("nc_identical", "nc_paraphrase", "nc_code_only", "nc_context_null")
DISPLAY_LABEL_SETS = {"letters": ("A", "B"), "numbers": ("1", "2")}

PARSER_POLICY = "strict_exact_code_v1"
SCORING_MODE = "first_and_full_target_margin_plus_strict_generation"

SYSTEM_PROMPT = (
    "You are an assistant working a software project's task queue. "
    "Follow the response format instructions exactly."
)
RO_SYSTEM_PROMPT = (
    "You are an assistant answering a planning survey about ways of "
    "working. Follow the response format instructions exactly."
)

# Capture/patch site vocabularies (plan §34; addendum E6).
AR_SITES = (
    "context_end", "option_a_end", "option_b_end", "menu_end",
    "response_instruction_start", "final_prompt_token",
)
RO_SITES = (
    "ro_context_end", "ro_option_a_end", "ro_option_b_end", "ro_menu_end",
    "ro_response_start", "ro_final_prompt_token",
)
RELATIVE_DEPTHS = (0.20, 0.35, 0.50, 0.65, 0.80, 0.95)


def advantage_target(context_strength: int) -> str:
    """Derived display field (addendum E4). Never stored independently."""
    if context_strength > 0:
        return "semantic_a"
    if context_strength < 0:
        return "semantic_b"
    return "neutral"


# Authored specs ------------------------------------------------------------

@dataclass(frozen=True)
class IncidentalSpec:
    """One surface skin. ``params`` fill templates and never change the
    tradeoff; ``incidental_split`` is frozen at authoring."""

    incidental_id: str
    incidental_split: str
    params: Mapping[str, str]


@dataclass(frozen=True)
class BindingSpec:
    binding_kind: str
    continuation_template_by_sem: Mapping[str, str]  # keys "a"/"b"
    validator_id: str
    max_new_tokens: int
    safety_class: str = "benign"
    # per-sem validator payload templates (filled with incidental params at
    # expansion) — validators read these instead of re-parsing option text
    validator_payload_by_sem: Mapping[str, Mapping[str, str]] | None = None


@dataclass(frozen=True)
class LadderStatement:
    """One context-ladder rung: signed strength, paraphrase family,
    statement template (advantage through scenario constraints only —
    addendum D wordlist guard)."""

    strength: int
    family: int  # paraphrase family 0..3
    template: str


@dataclass(frozen=True)
class ScenarioSpec:
    """A Phase 2 scenario/anchor/axis.

    ``option_template_by_sem`` maps {"a","b"} -> per-paraphrase templates
    (tuple of 2 for B-ARB3/PC; 1 elsewhere unless stated). ``framing_templates``
    is a tuple of paraphrase templates for the decision context.
    """

    scenario_id: str
    bank: str
    family: str
    contrast_axis: str
    semantic_a_id: str
    semantic_b_id: str
    framing_templates: tuple[str, ...]
    option_templates_a: tuple[str, ...]
    option_templates_b: tuple[str, ...]
    incidentals: tuple[IncidentalSpec, ...]
    binding: BindingSpec | None = None
    ladder: tuple[LadderStatement, ...] = ()
    ro_framing_templates: tuple[str, ...] = ()
    ro_option_templates_a: tuple[str, ...] = ()
    ro_option_templates_b: tuple[str, ...] = ()
    pc_family: str | None = None
    pc_expected_sem: str | None = None  # "a"/"b"
    nc_family: str | None = None
    canon_role: str | None = None  # "discovery" | "heldout" for B-CANON
    pcmech_difficulty: str | None = None  # "d1".."d4" on B-PC-MECH variants
    notes: str = ""

    def render(self, template: str, inc: IncidentalSpec) -> str:
        return normalize_newlines(template.format(**inc.params))


# Bank items ---------------------------------------------------------------

@dataclass
class BankItem:
    """One rendered prompt row (mutable until finalize_identity)."""

    # identity
    item_id: str | None = None
    semantic_key: str | None = None
    scientific_content_hash: str | None = None
    prompt_hash: str | None = None
    bank_version: str = BANK_VERSION
    schema_version: int = SCHEMA_VERSION
    # coordinates
    bank: str = ""
    family: str = ""
    channel: str = "AR"
    format_id: str = "F-SYM"
    scenario_id: str = ""
    contrast_axis: str = ""
    semantic_a_id: str = ""
    semantic_b_id: str = ""
    incidental_id: str = ""
    incidental_split: str = "train"
    prompt_subset: str = "frozen_only"  # "dev" rows usable pre-freeze
    # factors
    display_order: int = 0        # 0: semantic A's record first
    code_map_index: int = 0       # 0: pair[0] -> semantic A
    consequence_frame: str | None = None
    paraphrase_id: int = 0        # prompt paraphrase (menu paraphrase on B-MECH)
    context_strength: int = 0
    context_family: int | None = None  # ladder paraphrase family
    codebook_pair_id: str = ""
    codebook_reserved: bool = False
    display_label_set: str | None = None   # F-P1 only
    label_assignment: int | None = None    # F-P1 B-SURF: 0 first-displayed gets lower-rank label
    inline_code_assignment: int | None = None  # F-P1 B-SURF only
    reply_list_order: int | None = None        # F-P1 B-SURF only
    pcmech_difficulty: str | None = None
    # contract
    system_prompt: str = SYSTEM_PROMPT
    user_prompt: str = ""
    option_text_by_sem: dict[str, str] = field(default_factory=dict)
    response_code_by_sem: dict[str, str] = field(default_factory=dict)
    valid_codes_in_display_order: tuple[str, str] = ("", "")
    parser_policy: str = PARSER_POLICY
    scoring_mode: str = SCORING_MODE
    site_char_spans: dict[str, int] = field(default_factory=dict)
    # binding (E7)
    binding_kind: str | None = None
    continuation_by_sem: dict[str, str] | None = None
    validator_id: str | None = None
    validator_payload_by_sem: dict[str, dict[str, str]] | None = None
    binding_max_new_tokens: int = 0
    binding_safety_class: str | None = None
    # meta
    pc_family: str | None = None
    pc_expected_sem: str | None = None
    nc_family: str | None = None
    canon_role: str | None = None
    canon_context: str | None = None  # "neutral"|"favor_a"|"favor_b" (B-CANON)
    pair_key: str | None = None       # AR<->RO twin key
    notes: str | None = None

    def to_record(self) -> dict[str, Any]:
        rec = dataclasses.asdict(self)
        rec["advantage_target"] = advantage_target(self.context_strength)
        for k, v in list(rec.items()):
            if isinstance(v, tuple):
                rec[k] = list(v)
        return rec


def scientific_content_dict(item: BankItem) -> dict[str, Any]:
    """Exact hash-bound field set. No timestamps, no paths, no derived
    display fields; dict keys are strings already ("a"/"b")."""
    return {
        "schema_version": item.schema_version,
        "bank_version": item.bank_version,
        "bank": item.bank,
        "family": item.family,
        "channel": item.channel,
        "format_id": item.format_id,
        "scenario_id": item.scenario_id,
        "contrast_axis": item.contrast_axis,
        "semantic_a_id": item.semantic_a_id,
        "semantic_b_id": item.semantic_b_id,
        "incidental_id": item.incidental_id,
        "incidental_split": item.incidental_split,
        "display_order": item.display_order,
        "code_map_index": item.code_map_index,
        "consequence_frame": item.consequence_frame,
        "paraphrase_id": item.paraphrase_id,
        "context_strength": item.context_strength,
        "context_family": item.context_family,
        "codebook_pair_id": item.codebook_pair_id,
        "codebook_reserved": item.codebook_reserved,
        "display_label_set": item.display_label_set,
        "label_assignment": item.label_assignment,
        "inline_code_assignment": item.inline_code_assignment,
        "reply_list_order": item.reply_list_order,
        "pcmech_difficulty": item.pcmech_difficulty,
        "system_prompt": item.system_prompt,
        "user_prompt": item.user_prompt,
        "option_text_by_sem": dict(item.option_text_by_sem),
        "response_code_by_sem": dict(item.response_code_by_sem),
        "valid_codes_in_display_order": list(item.valid_codes_in_display_order),
        "parser_policy": item.parser_policy,
        "scoring_mode": item.scoring_mode,
        "binding_kind": item.binding_kind,
        "continuation_by_sem": (dict(item.continuation_by_sem)
                                if item.continuation_by_sem else None),
        "validator_id": item.validator_id,
        "validator_payload_by_sem": (
            {k: dict(v) for k, v in item.validator_payload_by_sem.items()}
            if item.validator_payload_by_sem else None),
        "binding_max_new_tokens": item.binding_max_new_tokens,
        "pc_expected_sem": item.pc_expected_sem,
        "canon_context": item.canon_context,
    }


def semantic_key(item: BankItem) -> str:
    frame = {"enacted": "fe", "hypothetical": "fh", None: "fx"}[item.consequence_frame]
    fmt = {"F-SYM": "fs", "F-P1": "fp", "F-COMMIT": "fc"}[item.format_id]
    parts = [
        item.bank.lower().replace("-", ""), item.scenario_id, item.incidental_id,
        item.channel.lower(), fmt, f"o{item.display_order}",
        f"c{item.code_map_index}", frame, f"p{item.paraphrase_id}",
        f"s{item.context_strength:+d}" if item.context_strength else "s0",
        item.codebook_pair_id,
    ]
    if item.pcmech_difficulty:
        parts.append(item.pcmech_difficulty)
    if item.display_label_set is not None:
        parts.append("ll" if item.display_label_set == "letters" else "ln")
    if item.label_assignment is not None:
        parts.append(f"la{item.label_assignment}")
    if item.inline_code_assignment is not None:
        parts.append(f"ic{item.inline_code_assignment}")
    if item.reply_list_order is not None:
        parts.append(f"rl{item.reply_list_order}")
    if item.canon_context is not None:
        parts.append({"neutral": "cn", "favor_a": "ca", "favor_b": "cb"}[item.canon_context])
    return "-".join(parts)


def pair_key(item: BankItem) -> str:
    """AR<->RO twin matching key: frame and paraphrase excluded, format
    excluded (the RO surface is disjoint by design)."""
    return "|".join([
        item.scenario_id, item.incidental_id, f"o{item.display_order}",
        f"c{item.code_map_index}",
    ])


def finalize_identity(item: BankItem) -> BankItem:
    item.pair_key = pair_key(item)
    item.semantic_key = semantic_key(item)
    item.scientific_content_hash = canonical_hash(scientific_content_dict(item))
    item.prompt_hash = canonical_hash(
        {"system": item.system_prompt, "user": item.user_prompt}
    )
    item.item_id = f"{item.semantic_key}-{item.scientific_content_hash[:12]}"
    return item


def sign_anchor_for(freeze_sha: str, scenario_id: str) -> int:
    """Analysis sign anchor (addendum E15): +1 keeps authored A as the
    positive direction, -1 flips. Derived, never authored."""
    return 1 if stable_seed("sign-anchor", freeze_sha, scenario_id) % 2 == 0 else -1
