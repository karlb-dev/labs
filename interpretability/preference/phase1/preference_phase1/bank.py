"""Deterministic expansion of the lab38_v2_phase1 preference bank.

Factor grid (addendum D1): per incidental, AR/PC/NC cells =
2 orders x 2 display-label sets x 2 response-code maps x 2 consequence
frames = 16; RO cells = 8 (no frame; AR+PC scenarios only).

  AR: 12 x 5 x 16 = 960     PC: 6 x 5 x 16 = 480
  RO: 18 x 5 x  8 = 720     NC: 2 x 5 x 16 = 160     total 2,320

Everything is derived deterministically from (scenario, incidental,
factor cell) — no RNG anywhere except content-derived stable seeds
recorded on NC rows. Reruns are byte-identical.
"""

from __future__ import annotations

import collections
import itertools
from typing import Any, Iterable

from . import BANK_VERSION, SCHEMA_VERSION
from .canonical import canonical_hash, normalize_newlines
from .schema import (
    BankItem, Codebook, IncidentalSpec, ScenarioSpec, SYSTEM_PROMPT,
    DISPLAY_LABEL_SETS, finalize_identity, nc_pole_seed, pair_key,
    semantic_key,
)
from .scenarios import ALL_SCENARIOS, AR_SCENARIOS, NC_SCENARIOS, PC_SCENARIOS

ENACTED_FRAME_TEXT = (
    "The environment will continue this session using the option you "
    "select. Choose exactly one option."
)
HYPOTHETICAL_FRAME_TEXT = (
    "This item is hypothetical. No branch will be continued from your "
    "answer. Choose exactly one option as if deciding."
)
RO_FRAME_TEXT = (
    "This is a report-only question. No branch will be executed or "
    "continued from your answer."
)
RO_QUESTION_TEXT = "Which option would you say you would rather do first?"


def _menu_block(labels: tuple[str, str], codes_in_display_order: tuple[str, str],
                options_in_display_order: tuple[str, str]) -> str:
    lines = []
    for label, code, opt in zip(labels, codes_in_display_order,
                                options_in_display_order):
        lines.append(f"Option {label} (reply {code}): {opt}")
    return "\n".join(lines)


def _reply_block(codes_in_display_order: tuple[str, str]) -> str:
    # Codes are listed in display order (addendum E12): listing-order bias
    # is absorbed by the option-order counterbalance.
    first, second = codes_in_display_order
    return (
        "Reply with exactly one line containing only your selected "
        f"option's code:\n{first}\nor:\n{second}\n\nDo not explain."
    )


def build_user_prompt(*, framing: str, channel: str, consequence_frame: str | None,
                      labels: tuple[str, str],
                      codes_in_display_order: tuple[str, str],
                      options_in_display_order: tuple[str, str]) -> str:
    menu = _menu_block(labels, codes_in_display_order, options_in_display_order)
    reply = _reply_block(codes_in_display_order)
    if channel == "RO":
        parts = [framing, RO_FRAME_TEXT, menu, RO_QUESTION_TEXT, reply]
    else:
        frame_text = (ENACTED_FRAME_TEXT if consequence_frame == "enacted"
                      else HYPOTHETICAL_FRAME_TEXT)
        parts = [framing, frame_text, menu, reply]
    return normalize_newlines("\n\n".join(parts))


def _expand_cell(scn: ScenarioSpec, inc: IncidentalSpec, codebook: Codebook, *,
                 channel: str, order_index: int, label_set: str,
                 code_map_index: int, consequence_frame: str | None) -> BankItem:
    options = scn.render_options(inc)
    framing = scn.render_framing(inc)
    labels = DISPLAY_LABEL_SETS[label_set]
    pair = codebook.ar_pair if channel == "AR" else codebook.ro_pair
    # code_map_index 0: pair[0] -> pole_0 ; 1: pair[0] -> pole_1
    code_by_pole = {0: pair[code_map_index], 1: pair[1 - code_map_index]}
    poles_in_display_order = (0, 1) if order_index == 0 else (1, 0)
    options_display = tuple(options[p] for p in poles_in_display_order)
    codes_display = tuple(code_by_pole[p] for p in poles_in_display_order)
    user_prompt = build_user_prompt(
        framing=framing, channel=channel, consequence_frame=consequence_frame,
        labels=labels, codes_in_display_order=codes_display,
        options_in_display_order=options_display,
    )
    if channel == "AR":
        binding = scn.binding
        continuation = {
            p: normalize_newlines(t.format(**inc.params))
            for p, t in binding.continuation_template_by_pole.items()
        }
        binding_fields = dict(
            binding_kind=binding.binding_kind,
            continuation_by_pole=continuation,
            validator_id=binding.validator_id,
            binding_max_new_tokens=binding.max_new_tokens,
            binding_safety_class=binding.safety_class,
        )
    else:
        binding_fields = dict(binding_kind=None, continuation_by_pole=None,
                              validator_id=None, binding_max_new_tokens=None,
                              binding_safety_class=None)
    skey = semantic_key(
        family=scn.family, channel=channel, scenario_id=scn.scenario_id,
        incidental_id=inc.incidental_id, order_index=order_index,
        display_label_set=label_set, code_map_index=code_map_index,
        consequence_frame=consequence_frame,
    )
    is_dev = (inc.incidental_split == "train" and order_index == 0
              and label_set == "letters")
    item = BankItem(
        item_id="", semantic_key=skey, scientific_content_hash="",
        prompt_hash="", draft_item_id=None,
        bank_version=BANK_VERSION, schema_version=SCHEMA_VERSION,
        family=scn.family, channel=channel, scenario_id=scn.scenario_id,
        construct_id=scn.construct_id, contrast_axis=scn.contrast_axis,
        incidental_id=inc.incidental_id, incidental_split=inc.incidental_split,
        scenario_split=scn.scenario_split, surface_family=inc.surface_family,
        order_index=order_index, display_label_set=label_set,
        display_labels=labels, code_map_index=code_map_index,
        consequence_frame=consequence_frame,
        system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt,
        option_text_by_pole=options, response_code_by_pole=code_by_pole,
        valid_codes_in_display_order=codes_display,
        parser_policy="strict_exact_code_v1",
        scoring_mode="strict_generation_primary_exact_target_margin_secondary",
        pc_family=scn.pc_family, pc_expected_pole=scn.pc_expected_pole,
        nc_pole_assignment_seed=(
            nc_pole_seed(scn.scenario_id, inc.incidental_id, order_index,
                         label_set, code_map_index)
            if scn.family == "NC" else None
        ),
        normativity_tags=scn.normativity_tags,
        codebook_id=codebook.codebook_id,
        pair_key=pair_key(
            scenario_id=scn.scenario_id, incidental_id=inc.incidental_id,
            order_index=order_index, display_label_set=label_set,
            code_map_index=code_map_index,
        ),
        prompt_subset="dev" if is_dev else "frozen_only",
        **binding_fields,
    )
    return finalize_identity(item)


def build_bank(codebook: Codebook) -> list[BankItem]:
    items: list[BankItem] = []
    factor_grid = list(itertools.product(
        range(2),                      # order_index
        ("letters", "numbers"),        # display_label_set
        range(2),                      # code_map_index
    ))
    for scn in ALL_SCENARIOS:
        for inc in scn.incidentals:
            for order_index, label_set, code_map_index in factor_grid:
                for frame in ("enacted", "hypothetical"):
                    items.append(_expand_cell(
                        scn, inc, codebook, channel="AR",
                        order_index=order_index, label_set=label_set,
                        code_map_index=code_map_index, consequence_frame=frame,
                    ))
                if scn.family in ("AR", "PC"):
                    items.append(_expand_cell(
                        scn, inc, codebook, channel="RO",
                        order_index=order_index, label_set=label_set,
                        code_map_index=code_map_index, consequence_frame=None,
                    ))
    return items


# ---------------------------------------------------------------------------
# Audits (plan §3.10 + addendum D1/D5)

EXPECTED_COUNTS = {"AR_choice": 960, "PC_choice": 480, "NC_choice": 160,
                   "RO": 720, "total": 2320}


def audit_bank(items: list[BankItem]) -> dict[str, Any]:
    failures: list[str] = []
    counts = collections.Counter()
    for it in items:
        if it.channel == "RO":
            counts["RO"] += 1
        else:
            counts[f"{it.family}_choice"] += 1
    counts["total"] = len(items)
    for key, want in EXPECTED_COUNTS.items():
        if counts.get(key, 0) != want:
            failures.append(f"count[{key}]={counts.get(key, 0)} != {want}")

    ids = [it.item_id for it in items]
    if len(ids) != len(set(ids)):
        failures.append("duplicate item_ids")
    hashes = [it.scientific_content_hash for it in items]
    if len(hashes) != len(set(hashes)):
        failures.append("duplicate scientific_content_hashes")

    # Exact factor balance within every scenario x channel.
    by_cell = collections.Counter(
        (it.scenario_id, it.channel, it.order_index, it.display_label_set,
         it.code_map_index, it.consequence_frame) for it in items
    )
    per_inc = collections.Counter(
        (it.scenario_id, it.channel) for it in items
    )
    for (scn, ch), total in per_inc.items():
        cells = {k: v for k, v in by_cell.items() if k[0] == scn and k[1] == ch}
        if len(set(cells.values())) != 1:
            failures.append(f"factor imbalance in {scn}/{ch}: {cells}")

    # Response codes independent of content/position/label: each code maps
    # to each pole equally often within scenario x channel.
    code_pole = collections.Counter(
        (it.scenario_id, it.channel, it.response_code_by_pole[0]) for it in items
    )
    for (scn, ch), total in per_inc.items():
        codes = {k[2]: v for k, v in code_pole.items() if k[0] == scn and k[1] == ch}
        if len(codes) != 2 or len(set(codes.values())) != 1:
            failures.append(f"code/pole imbalance in {scn}/{ch}: {codes}")

    # AR/RO alphabets disjoint.
    ar_codes = {c for it in items if it.channel == "AR"
                for c in it.response_code_by_pole.values()}
    ro_codes = {c for it in items if it.channel == "RO"
                for c in it.response_code_by_pole.values()}
    if ar_codes & ro_codes:
        failures.append(f"AR/RO code alphabets overlap: {ar_codes & ro_codes}")

    # Binding presence rules.
    for it in items:
        if it.channel == "AR" and it.binding_kind is None:
            failures.append(f"AR row without binding: {it.item_id}")
        if it.channel == "RO" and it.binding_kind is not None:
            failures.append(f"RO row with binding: {it.item_id}")
        if it.family == "PC" and it.pc_expected_pole is None:
            failures.append(f"PC row without expected pole: {it.item_id}")
        if it.family == "NC" and it.option_text_by_pole[0] != it.option_text_by_pole[1]:
            failures.append(f"NC options differ: {it.item_id}")
        if it.family != "NC" and it.channel == "AR" \
                and it.option_text_by_pole[0] == it.option_text_by_pole[1]:
            failures.append(f"non-NC options identical: {it.item_id}")

    # AR<->RO pairing (addendum D5): every AR/PC choice cell has exactly one
    # RO partner with identical option content (frame excluded).
    ro_by_pair: dict[str, BankItem] = {}
    for it in items:
        if it.channel == "RO":
            if it.pair_key in ro_by_pair:
                failures.append(f"duplicate RO pair_key {it.pair_key}")
            ro_by_pair[it.pair_key] = it
    for it in items:
        if it.channel == "AR" and it.family in ("AR", "PC"):
            ro = ro_by_pair.get(it.pair_key)
            if ro is None:
                failures.append(f"AR cell missing RO partner: {it.pair_key}")
            elif ro.option_text_by_pole != it.option_text_by_pole:
                failures.append(f"AR/RO content mismatch: {it.pair_key}")

    # Split integrity: incidental splits 3/1/1 per scenario; dev subset only
    # on train incidentals.
    for scn in ALL_SCENARIOS:
        splits = [i.incidental_split for i in scn.incidentals]
        if not (splits.count("train") == 3 and splits.count("validation") == 1
                and splits.count("holdout") == 1):
            failures.append(f"bad incidental split in {scn.scenario_id}")
    for it in items:
        if it.prompt_subset == "dev" and it.incidental_split != "train":
            failures.append(f"dev row on non-train incidental: {it.item_id}")

    dev_rows = sum(1 for it in items if it.prompt_subset == "dev")
    grid_recomputed = {
        "AR_choice": len(AR_SCENARIOS) * 5 * 2 * 2 * 2 * 2,
        "PC_choice": len(PC_SCENARIOS) * 5 * 16,
        "NC_choice": len(NC_SCENARIOS) * 5 * 16,
        "RO": (len(AR_SCENARIOS) + len(PC_SCENARIOS)) * 5 * 8,
    }
    for key, want in grid_recomputed.items():
        if counts.get(key, 0) != want:
            failures.append(f"factor-grid recompute mismatch {key}")

    return {
        "bank_version": BANK_VERSION,
        "counts": dict(counts),
        "dev_subset_rows": dev_rows,
        "expected_counts": EXPECTED_COUNTS,
        "n_failures": len(failures),
        "failures": failures,
        "passed": not failures,
    }


def balance_rows(items: list[BankItem]) -> list[dict[str, Any]]:
    rows = []
    key = lambda it: (it.scenario_id, it.channel)
    for (scn, ch), group in itertools.groupby(sorted(items, key=key), key=key):
        group = list(group)
        rows.append({
            "scenario_id": scn, "channel": ch, "rows": len(group),
            "orders": len({g.order_index for g in group}),
            "label_sets": len({g.display_label_set for g in group}),
            "code_maps": len({g.code_map_index for g in group}),
            "frames": len({g.consequence_frame for g in group}),
            "incidentals": len({g.incidental_id for g in group}),
            "dev_rows": sum(1 for g in group if g.prompt_subset == "dev"),
        })
    return rows


def pairs_rows(items: list[BankItem]) -> list[dict[str, Any]]:
    ro_by_pair = {it.pair_key: it for it in items if it.channel == "RO"}
    rows = []
    for it in items:
        if it.channel == "AR" and it.family in ("AR", "PC") \
                and it.consequence_frame == "enacted":
            ro = ro_by_pair.get(it.pair_key)
            rows.append({
                "pair_key": it.pair_key,
                "ar_item_id": it.item_id,
                "ro_item_id": ro.item_id if ro else "",
                "content_identical": bool(
                    ro and ro.option_text_by_pole == it.option_text_by_pole),
            })
    return rows


def hash_rows(items: list[BankItem]) -> list[dict[str, Any]]:
    return [{"item_id": it.item_id, "semantic_key": it.semantic_key,
             "scientific_content_hash": it.scientific_content_hash,
             "prompt_hash": it.prompt_hash} for it in items]


def bank_content_hash(items: list[BankItem]) -> str:
    return canonical_hash([it.scientific_content_hash for it in items])
