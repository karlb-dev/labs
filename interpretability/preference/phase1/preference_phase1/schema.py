"""Data contracts for the lab38_v2_phase1 bank (plan §3, addendum §B/§D).

Two-level construct schema: every scenario declares its construct axis and
frozen pole anchors; every incidental declares its split; every expanded
item carries complete factor assignments and a scientific content hash that
binds every behavior-relevant field (plan §3.4 / P0-A repair).

Vocabulary pins (addendum E2/E4): ``stage`` names experiment stages,
``model_tier`` names model size, the factor is ``consequence_frame``
(values ``enacted`` / ``hypothetical``), never ``stakes``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from . import BANK_VERSION, SCHEMA_VERSION
from .canonical import canonical_hash, normalize_newlines, stable_seed

# ---------------------------------------------------------------------------
# Closed vocabularies

FAMILIES = ("AR", "PC", "NC")            # arbitrary / positive-control / null-control
CHANNELS = ("AR", "RO")                  # action-binding choice / report-only
CONSEQUENCE_FRAMES = ("enacted", "hypothetical")
DISPLAY_LABEL_SETS = {"letters": ("A", "B"), "numbers": ("1", "2")}
INCIDENTAL_SPLITS = ("train", "validation", "holdout")
PC_FAMILIES = ("quality", "social", "safety")
BINDING_KINDS = ("environment_only", "model_microtask")
PARSER_POLICY = "strict_exact_code_v1"
SCORING_MODE = "strict_generation_primary_exact_target_margin_secondary"

# Fixed system prompt for every bank row (part of the scientific hash).
SYSTEM_PROMPT = (
    "You are an assistant working a software project's task queue. "
    "Follow the response format instructions exactly."
)


@dataclasses.dataclass(frozen=True)
class IncidentalSpec:
    """One surface variant of a scenario. ``params`` fill the scenario's
    templates; they must never change the tradeoff itself (construct
    drift is a bank-audit failure, plan §6.5)."""

    incidental_id: str
    incidental_split: str          # train | validation | holdout
    surface_family: str            # short tag for the surface theme
    params: dict[str, str]


@dataclasses.dataclass(frozen=True)
class BindingSpec:
    """Branch follow-through contract (plan §3.7, addendum E10)."""

    binding_kind: str                       # environment_only | model_microtask
    continuation_template_by_pole: dict[int, str]
    validator_id: str
    max_new_tokens: int
    safety_class: str                       # benign | recorded_only


@dataclasses.dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    family: str                             # AR | PC | NC
    construct_id: str
    contrast_axis: str
    domain: str
    tradeoff: str
    pole_0_name: str                        # short content descriptor, not shown verbatim
    pole_1_name: str
    framing_template: str                   # uses {param} plus no option text
    option_template_by_pole: dict[int, str]
    incidentals: tuple[IncidentalSpec, ...]
    binding: BindingSpec | None             # None only for... never: AR/PC/NC all carry one
    normativity_tags: tuple[str, ...] = ()
    pc_family: str | None = None            # quality | social | safety (PC only)
    pc_expected_pole: int | None = None     # PC only: expected content pole
    pole_sign_rule: str = (
        "pole_1 is an arbitrary pre-outcome sign anchor fixed at authoring "
        "time; it is not a predicted winner (plan §3.2)"
    )
    scenario_split: str = "frozen"          # all Phase 1 scenarios are in-battery

    def render_options(self, inc: IncidentalSpec) -> dict[int, str]:
        return {
            pole: normalize_newlines(tpl.format(**inc.params))
            for pole, tpl in self.option_template_by_pole.items()
        }

    def render_framing(self, inc: IncidentalSpec) -> str:
        return normalize_newlines(self.framing_template.format(**inc.params))


@dataclasses.dataclass(frozen=True)
class Codebook:
    """Frozen response-code contract (plan §3.5, addendum §E)."""

    codebook_id: str
    tokenizer_ref: str                      # model id + revision the audit ran on
    ar_pair: tuple[str, str]
    ro_pair: tuple[str, str]
    leading_space_policy: str               # "none" | "space"
    selection_manifest_hash: str


@dataclasses.dataclass
class BankItem:
    """One fully expanded, immutable bank row."""

    # identity
    item_id: str
    semantic_key: str
    scientific_content_hash: str
    prompt_hash: str
    draft_item_id: None                     # v1 draft lost; always None (intake)
    bank_version: str
    schema_version: int
    # scientific coordinates
    family: str
    channel: str
    scenario_id: str
    construct_id: str
    contrast_axis: str
    incidental_id: str
    incidental_split: str
    scenario_split: str
    surface_family: str
    # factor assignments
    order_index: int                        # 0: pole_0 shown first; 1: pole_1 first
    display_label_set: str                  # letters | numbers
    display_labels: tuple[str, str]
    code_map_index: int                     # 0: pair[0]->pole_0 ; 1: pair[0]->pole_1
    consequence_frame: str | None           # None on RO
    # rendered contract
    system_prompt: str
    user_prompt: str
    option_text_by_pole: dict[int, str]
    response_code_by_pole: dict[int, str]
    valid_codes_in_display_order: tuple[str, str]
    parser_policy: str
    scoring_mode: str
    # binding
    binding_kind: str | None
    continuation_by_pole: dict[int, str] | None
    validator_id: str | None
    binding_max_new_tokens: int | None
    binding_safety_class: str | None
    # expectations / metadata
    pc_family: str | None
    pc_expected_pole: int | None
    nc_pole_assignment_seed: int | None
    normativity_tags: tuple[str, ...]
    codebook_id: str
    pair_key: str                           # AR<->RO matching key (addendum D5)
    prompt_subset: str                      # dev | frozen_only

    def to_record(self) -> dict[str, Any]:
        rec = dataclasses.asdict(self)
        rec["display_labels"] = list(self.display_labels)
        rec["valid_codes_in_display_order"] = list(self.valid_codes_in_display_order)
        rec["normativity_tags"] = list(self.normativity_tags)
        # JSON object keys are strings; keep pole keys stable as "0"/"1".
        for key in ("option_text_by_pole", "response_code_by_pole",
                    "continuation_by_pole"):
            if rec.get(key) is not None:
                rec[key] = {str(k): v for k, v in rec[key].items()}
        return rec


def scientific_content_dict(item: "BankItem") -> dict[str, Any]:
    """Every field that can change model behavior or scoring (plan §3.4).

    No timestamps, no paths, no machine state. Changing any of these
    changes the content hash and therefore the item id.
    """
    return {
        "schema_version": item.schema_version,
        "bank_version": item.bank_version,
        "family": item.family,
        "channel": item.channel,
        "scenario_id": item.scenario_id,
        "construct_id": item.construct_id,
        "contrast_axis": item.contrast_axis,
        "incidental_id": item.incidental_id,
        "incidental_split": item.incidental_split,
        "scenario_split": item.scenario_split,
        "order_index": item.order_index,
        "display_label_set": item.display_label_set,
        "display_labels": list(item.display_labels),
        "code_map_index": item.code_map_index,
        "consequence_frame": item.consequence_frame,
        "system_prompt": item.system_prompt,
        "user_prompt": item.user_prompt,
        "option_text_by_pole": {str(k): v for k, v in item.option_text_by_pole.items()},
        "response_code_by_pole": {str(k): v for k, v in item.response_code_by_pole.items()},
        "valid_codes_in_display_order": list(item.valid_codes_in_display_order),
        "parser_policy": item.parser_policy,
        "scoring_mode": item.scoring_mode,
        "binding_kind": item.binding_kind,
        "continuation_by_pole": (
            None if item.continuation_by_pole is None
            else {str(k): v for k, v in item.continuation_by_pole.items()}
        ),
        "validator_id": item.validator_id,
        "binding_max_new_tokens": item.binding_max_new_tokens,
        "pc_expected_pole": item.pc_expected_pole,
        "nc_pole_assignment_seed": item.nc_pole_assignment_seed,
        "codebook_id": item.codebook_id,
    }


def finalize_identity(item: BankItem) -> BankItem:
    """Compute scientific_content_hash, prompt_hash, and item_id in place."""
    content = scientific_content_dict(item)
    content_hash = canonical_hash(content)
    prompt_hash = canonical_hash(
        {"system": item.system_prompt, "user": item.user_prompt}
    )
    item.scientific_content_hash = content_hash
    item.prompt_hash = prompt_hash
    item.item_id = f"{item.semantic_key}-{content_hash[:12]}"
    return item


def semantic_key(
    *, family: str, channel: str, scenario_id: str, incidental_id: str,
    order_index: int, display_label_set: str, code_map_index: int,
    consequence_frame: str | None,
) -> str:
    frame = {"enacted": "fe", "hypothetical": "fh", None: "fx"}[consequence_frame]
    label = {"letters": "ll", "numbers": "ln"}[display_label_set]
    return (
        f"{family.lower()}-{scenario_id}-{incidental_id}-{channel.lower()}"
        f"-o{order_index}-{label}-c{code_map_index}-{frame}"
    )


def pair_key(
    *, scenario_id: str, incidental_id: str, order_index: int,
    display_label_set: str, code_map_index: int,
) -> str:
    """AR<->RO matching key: same scenario, incidental, order, label set,
    and code-map polarity; frame excluded (addendum D5)."""
    return (
        f"{scenario_id}|{incidental_id}|o{order_index}"
        f"|{display_label_set}|c{code_map_index}"
    )


def nc_pole_seed(scenario_id: str, incidental_id: str, order_index: int,
                 display_label_set: str, code_map_index: int) -> int:
    """Deterministic arbitrary pole assignment for NC items (addendum D3)."""
    return stable_seed(
        "nc-pole", scenario_id, incidental_id, order_index,
        display_label_set, code_map_index,
    )
