"""Deterministic bank v3 expansion + audits (plan Part IV; addendum D).

No RNG anywhere: every row is a pure function of authored content and
the codebook manifest. Reruns are byte-identical. Counts of record:

    B-SURF      4 x 8 x (32 F-P1 + 4 F-SYM)                 = 1,152
    B-ARB3      12 x 24 x (2 order x 2 cmap x 2 frame x 2 para) = 4,608
    B-MECH      3 x (32 x 40 + 8 x 40 reserved)             = 4,800
    B-CANON     6 x 8 x (3 ctx x 2 x 2)                     =   576
    B-PC        6 x 5 x 16                                  =   480
    B-PC-MECH   4 difficulty x 32 x (5 s x 2 x 2)           = 2,560
    B-NC        2x6x16 + 2x6x16 + 1x6x16 + 1x8x20           =   640
    RO-DISJOINT 12x24x(2x2x2) + 3 anchors x (64 + 64 rsv)   = 2,688
    F-P1-CONT   4 x 6 x (2 order x 2 cmap x 2 frame x 2 fam) =  384
    B-DEV       6 x 4 x (8 F-SYM + 8 F-P1 + 2 F-COMMIT)     =   432

Two pinned departures from the addendum-D table, both raises (permitted
pre-freeze; never lower): (1) RO-DISJOINT includes the Section-G coupling
receivers explicitly (64 primary + 64 reserved per anchor), which the D
arithmetic undercounted; (2) B-ARB3 runs 24 incidentals (12/6/6), raised
from 16 by the power simulation (plan §32: strict-choice power at the
0.10 SESOI was 0.45 at 16 incidentals, 0.87 at 24, under the E16 exact
sign-flip + Holm-12 primary). RO twins follow at 24.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable

from . import BANK_VERSION
from .canonical import canonical_hash
from .codebooks import CodebookFamilies, rotation_pair_for
from .content_aux import canon_contexts
from .formats import (ENACTED_FRAME_TEXT, HYPOTHETICAL_FRAME_TEXT,
                      order_site_spans, render_fcommit, render_fp1,
                      render_fsym, render_ro)
from .schema import (BankItem, IncidentalSpec, RO_SYSTEM_PROMPT,
                     ScenarioSpec, finalize_identity)
from .scenarios import (ALL_NC_SCENARIOS, ARB3_SCENARIOS, CANON_SCENARIOS,
                        DEV_SCENARIOS, MECH_SCENARIOS, NC_CTXNULL,
                        PCMECH_SCENARIOS, PC_SCENARIOS, SURF_SCENARIOS)

EXPECTED_COUNTS = {
    "B-SURF": 1152, "B-ARB3": 4608, "B-MECH": 4800, "B-CANON": 576,
    "B-PC": 480, "B-PC-MECH": 2560, "B-NC": 640, "RO-DISJOINT": 2688,
    "F-P1-CONT": 384, "B-DEV": 432,
}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())

_FRAME_TEXT = {"enacted": ENACTED_FRAME_TEXT,
               "hypothetical": HYPOTHETICAL_FRAME_TEXT}

CONTINUITY_SCENARIO_IDS = ("arb_setup", "arb_execmode", "arb_docsection",
                           "nc_ident_deploy")


def _ladder_map(scn: ScenarioSpec) -> dict[tuple[int, int], str]:
    return {(st.family, st.strength): st.template for st in scn.ladder}


def _context_family_for(index: int) -> int:
    """Ladder paraphrase family balanced within splits (E5): incidentals
    cycle families 0..3 positionally, which is balanced because split
    sizes are multiples of 4."""
    return index % 4


def _fill_payload(scn: ScenarioSpec, inc: IncidentalSpec) -> dict | None:
    if not scn.binding or not scn.binding.validator_payload_by_sem:
        return None
    return {
        sem: {k: scn.render(v, inc) for k, v in payload.items()}
        for sem, payload in scn.binding.validator_payload_by_sem.items()
    }


def _base_item(scn: ScenarioSpec, inc: IncidentalSpec, bank: str,
               pair, code_map_index: int) -> BankItem:
    codes = {"a": pair.codes[code_map_index],
             "b": pair.codes[1 - code_map_index]}
    item = BankItem(
        bank=bank, family=scn.family, scenario_id=scn.scenario_id,
        contrast_axis=scn.contrast_axis,
        semantic_a_id=scn.semantic_a_id, semantic_b_id=scn.semantic_b_id,
        incidental_id=inc.incidental_id,
        incidental_split=inc.incidental_split,
        code_map_index=code_map_index,
        codebook_pair_id=pair.pair_id,
        codebook_reserved=(pair.role == "reserved_transfer"),
        response_code_by_sem=codes,
        pc_family=scn.pc_family, pc_expected_sem=scn.pc_expected_sem,
        nc_family=scn.nc_family, canon_role=scn.canon_role,
        pcmech_difficulty=scn.pcmech_difficulty,
    )
    if scn.binding:
        item.binding_kind = scn.binding.binding_kind
        item.continuation_by_sem = {
            sem: scn.render(tpl, inc)
            for sem, tpl in scn.binding.continuation_template_by_sem.items()
        }
        item.validator_id = scn.binding.validator_id
        item.validator_payload_by_sem = _fill_payload(scn, inc)
        item.binding_max_new_tokens = scn.binding.max_new_tokens
        item.binding_safety_class = scn.binding.safety_class
    return item


def _finish_ar_fsym(item: BankItem, scn: ScenarioSpec, inc: IncidentalSpec,
                    *, display_order: int, paraphrase_id: int,
                    consequence_frame: str | None,
                    context_statement: str | None) -> BankItem:
    pi = paraphrase_id if len(scn.framing_templates) > 1 else 0
    framing = scn.render(scn.framing_templates[pi], inc)
    opt_a = scn.render(scn.option_templates_a[pi if len(scn.option_templates_a) > 1 else 0], inc)
    opt_b = scn.render(scn.option_templates_b[pi if len(scn.option_templates_b) > 1 else 0], inc)
    item.format_id = "F-SYM"
    item.display_order = display_order
    item.paraphrase_id = paraphrase_id
    item.consequence_frame = consequence_frame
    item.option_text_by_sem = {"a": opt_a, "b": opt_b}
    first_sem, second_sem = ("a", "b") if display_order == 0 else ("b", "a")
    texts = {"a": opt_a, "b": opt_b}
    prompt, spans = render_fsym(
        framing=framing, context_statement=context_statement,
        frame_text=_FRAME_TEXT.get(consequence_frame),
        first_code=item.response_code_by_sem[first_sem],
        first_text=texts[first_sem],
        second_code=item.response_code_by_sem[second_sem],
        second_text=texts[second_sem],
    )
    item.user_prompt = prompt
    item.site_char_spans = order_site_spans(spans, display_order=display_order,
                                            channel="AR")
    item.valid_codes_in_display_order = (
        item.response_code_by_sem[first_sem],
        item.response_code_by_sem[second_sem],
    )
    return finalize_identity(item)


def _build_arb3(families: CodebookFamilies) -> list[BankItem]:
    items = []
    for scn in ARB3_SCENARIOS:
        for idx, inc in enumerate(scn.incidentals):
            pair = rotation_pair_for(families, "AR", scn.scenario_id, idx,
                                     inc.incidental_split)
            for order, cmap, frame, para in itertools.product(
                    range(2), range(2), ("enacted", "hypothetical"), range(2)):
                item = _base_item(scn, inc, "B-ARB3", pair, cmap)
                items.append(_finish_ar_fsym(
                    item, scn, inc, display_order=order, paraphrase_id=para,
                    consequence_frame=frame, context_statement=None))
    return items


def _build_mech(families: CodebookFamilies) -> list[BankItem]:
    items = []
    for scn in MECH_SCENARIOS:
        ladder = _ladder_map(scn)
        for idx, inc in enumerate(scn.incidentals):
            fam = _context_family_for(idx)
            primary = rotation_pair_for(families, "AR", scn.scenario_id, idx,
                                        inc.incidental_split)
            pairs = [primary]
            if inc.incidental_split == "holdout":
                pairs.append(families.reserved_pair("AR"))
            for pair in pairs:
                for strength, order, cmap, para in itertools.product(
                        (-2, -1, 0, 1, 2), range(2), range(2), range(2)):
                    item = _base_item(scn, inc, "B-MECH", pair, cmap)
                    item.context_strength = strength
                    item.context_family = fam
                    ctx = scn.render(ladder[(fam, strength)], inc)
                    items.append(_finish_ar_fsym(
                        item, scn, inc, display_order=order,
                        paraphrase_id=para, consequence_frame="enacted",
                        context_statement=ctx))
    return items


def _build_pcmech(families: CodebookFamilies) -> list[BankItem]:
    items = []
    for scn in PCMECH_SCENARIOS:
        ladder = _ladder_map(scn)
        for idx, inc in enumerate(scn.incidentals):
            fam = _context_family_for(idx)
            pair = rotation_pair_for(families, "AR", scn.scenario_id, idx,
                                     inc.incidental_split)
            for strength, order, cmap in itertools.product(
                    (-2, -1, 0, 1, 2), range(2), range(2)):
                item = _base_item(scn, inc, "B-PC-MECH", pair, cmap)
                item.context_strength = strength
                item.context_family = fam
                ctx = scn.render(ladder[(fam, strength)], inc)
                items.append(_finish_ar_fsym(
                    item, scn, inc, display_order=order, paraphrase_id=0,
                    consequence_frame="enacted", context_statement=ctx))
    return items


def _build_canon(families: CodebookFamilies) -> list[BankItem]:
    items = []
    for scn in CANON_SCENARIOS:
        contexts = canon_contexts(scn)
        for idx, inc in enumerate(scn.incidentals):
            pair = rotation_pair_for(families, "AR", scn.scenario_id, idx,
                                     inc.incidental_split)
            for ctx_key, order, cmap in itertools.product(
                    ("neutral", "favor_a", "favor_b"), range(2), range(2)):
                item = _base_item(scn, inc, "B-CANON", pair, cmap)
                item.canon_context = ctx_key
                item.context_strength = {"neutral": 0, "favor_a": 1,
                                         "favor_b": -1}[ctx_key]
                ctx = scn.render(contexts[ctx_key], inc)
                items.append(_finish_ar_fsym(
                    item, scn, inc, display_order=order, paraphrase_id=0,
                    consequence_frame="enacted", context_statement=ctx))
    return items


def _build_pc(families: CodebookFamilies) -> list[BankItem]:
    items = []
    for scn in PC_SCENARIOS:
        for idx, inc in enumerate(scn.incidentals):
            pair = rotation_pair_for(families, "AR", scn.scenario_id, idx,
                                     inc.incidental_split)
            for order, cmap, frame, para in itertools.product(
                    range(2), range(2), ("enacted", "hypothetical"), range(2)):
                item = _base_item(scn, inc, "B-PC", pair, cmap)
                items.append(_finish_ar_fsym(
                    item, scn, inc, display_order=order, paraphrase_id=para,
                    consequence_frame=frame, context_statement=None))
    return items


def _build_nc(families: CodebookFamilies) -> list[BankItem]:
    items = []
    for scn in ALL_NC_SCENARIOS:
        if scn.scenario_id == "nc_ctxnull":
            ladder = _ladder_map(scn)
            for idx, inc in enumerate(scn.incidentals):
                fam = _context_family_for(idx)
                pair = rotation_pair_for(families, "AR", scn.scenario_id,
                                         idx, inc.incidental_split)
                for strength, order, cmap in itertools.product(
                        (-2, -1, 0, 1, 2), range(2), range(2)):
                    item = _base_item(scn, inc, "B-NC", pair, cmap)
                    item.context_strength = strength
                    item.context_family = fam
                    ctx = scn.render(ladder[(fam, strength)], inc)
                    items.append(_finish_ar_fsym(
                        item, scn, inc, display_order=order, paraphrase_id=0,
                        consequence_frame="enacted", context_statement=ctx))
        else:
            for idx, inc in enumerate(scn.incidentals):
                pair = rotation_pair_for(families, "AR", scn.scenario_id,
                                         idx, inc.incidental_split)
                for order, cmap, frame, para in itertools.product(
                        range(2), range(2), ("enacted", "hypothetical"),
                        range(2)):
                    item = _base_item(scn, inc, "B-NC", pair, cmap)
                    items.append(_finish_ar_fsym(
                        item, scn, inc, display_order=order,
                        paraphrase_id=para, consequence_frame=frame,
                        context_statement=None))
    return items


def _build_surf(families: CodebookFamilies) -> list[BankItem]:
    """B-SURF factorial (addendum E3): per template-skin, F-P1 contributes
    2^5 cells (order x label-assignment x inline-code x reply-list x
    label-family) and F-SYM contributes 2^2 (order x code map)."""
    items = []
    for scn in SURF_SCENARIOS:
        for idx, inc in enumerate(scn.incidentals):
            pair = rotation_pair_for(families, "AR", scn.scenario_id, idx,
                                     inc.incidental_split)
            framing = scn.render(scn.framing_templates[0], inc)
            twin_x = scn.render(scn.option_templates_a[0], inc)
            twin_y = scn.render(scn.option_templates_b[0], inc)
            # F-SYM cells
            for order, cmap in itertools.product(range(2), range(2)):
                item = _base_item(scn, inc, "B-SURF", pair, cmap)
                items.append(_finish_ar_fsym(
                    item, scn, inc, display_order=order, paraphrase_id=0,
                    consequence_frame="enacted", context_statement=None))
            # F-P1 cells
            for order, la, ic, rl, famname in itertools.product(
                    range(2), range(2), range(2), range(2),
                    ("letters", "numbers")):
                item = _base_item(scn, inc, "B-SURF", pair, ic)
                item.format_id = "F-P1"
                item.display_order = order
                item.display_label_set = famname
                item.label_assignment = la
                item.inline_code_assignment = ic
                item.reply_list_order = rl
                item.consequence_frame = "enacted"
                item.option_text_by_sem = {"a": twin_x, "b": twin_y}
                first_sem, second_sem = ("a", "b") if order == 0 else ("b", "a")
                texts = {"a": twin_x, "b": twin_y}
                base_labels = {"letters": ("A", "B"), "numbers": ("1", "2")}[famname]
                labels = base_labels if la == 0 else (base_labels[1], base_labels[0])
                codes_disp = (item.response_code_by_sem[first_sem],
                              item.response_code_by_sem[second_sem])
                reply_codes = codes_disp if rl == 0 else (codes_disp[1], codes_disp[0])
                prompt, spans = render_fp1(
                    framing=framing, frame_text=ENACTED_FRAME_TEXT,
                    labels=labels, codes_in_display_order=codes_disp,
                    options_in_display_order=(texts[first_sem], texts[second_sem]),
                    reply_codes_in_list_order=reply_codes,
                )
                item.user_prompt = prompt
                item.site_char_spans = spans
                item.valid_codes_in_display_order = codes_disp
                items.append(finalize_identity(item))
    return items


def _ro_rows_for(scn: ScenarioSpec, families: CodebookFamilies,
                 *, incidentals: Iterable[tuple[int, IncidentalSpec]],
                 include_reserved: bool) -> list[BankItem]:
    items = []
    for idx, inc in incidentals:
        primary = rotation_pair_for(families, "RO", scn.scenario_id, idx,
                                    inc.incidental_split)
        pairs = [primary]
        if include_reserved and inc.incidental_split == "holdout":
            pairs.append(families.reserved_pair("RO"))
        for pair in pairs:
            for order, cmap, para in itertools.product(range(2), range(2),
                                                       range(2)):
                item = _base_item(scn, inc, "RO-DISJOINT", pair, cmap)
                item.channel = "RO"
                item.format_id = "F-SYM"
                item.system_prompt = RO_SYSTEM_PROMPT
                item.display_order = order
                item.paraphrase_id = para
                item.consequence_frame = None
                # RO rows never execute (Phase 1 invariant; addendum E7)
                item.binding_kind = None
                item.continuation_by_sem = None
                item.validator_id = None
                item.validator_payload_by_sem = None
                item.binding_max_new_tokens = 0
                item.binding_safety_class = None
                ro_framing = scn.render(scn.ro_framing_templates[para], inc)
                opt_a = scn.render(scn.ro_option_templates_a[para], inc)
                opt_b = scn.render(scn.ro_option_templates_b[para], inc)
                item.option_text_by_sem = {"a": opt_a, "b": opt_b}
                first_sem, second_sem = ("a", "b") if order == 0 else ("b", "a")
                texts = {"a": opt_a, "b": opt_b}
                prompt, spans = render_ro(
                    ro_framing=ro_framing,
                    first_code=item.response_code_by_sem[first_sem],
                    first_text=texts[first_sem],
                    second_code=item.response_code_by_sem[second_sem],
                    second_text=texts[second_sem],
                )
                item.user_prompt = prompt
                item.site_char_spans = order_site_spans(
                    spans, display_order=order, channel="RO")
                item.valid_codes_in_display_order = (
                    item.response_code_by_sem[first_sem],
                    item.response_code_by_sem[second_sem],
                )
                items.append(finalize_identity(item))
    return items


def _build_ro(families: CodebookFamilies) -> list[BankItem]:
    items = []
    for scn in ARB3_SCENARIOS:
        items.extend(_ro_rows_for(
            scn, families,
            incidentals=list(enumerate(scn.incidentals)),
            include_reserved=False))
    for scn in MECH_SCENARIOS:
        holdouts = [(i, inc) for i, inc in enumerate(scn.incidentals)
                    if inc.incidental_split == "holdout"]
        items.extend(_ro_rows_for(scn, families, incidentals=holdouts,
                                  include_reserved=True))
    return items


def _build_continuity(families: CodebookFamilies) -> list[BankItem]:
    """F-P1 continuity arm (addendum D): Phase 2 twins of the strongest
    Phase 1 cells, rendered in the Phase 1 clone format."""
    from .scenarios import scenario_by_id
    items = []
    for sid in CONTINUITY_SCENARIO_IDS:
        scn = scenario_by_id(sid)
        for idx, inc in list(enumerate(scn.incidentals))[:6]:
            pair = rotation_pair_for(families, "AR", scn.scenario_id, idx,
                                     inc.incidental_split)
            framing = scn.render(scn.framing_templates[0], inc)
            opt_a = scn.render(scn.option_templates_a[0], inc)
            opt_b = scn.render(scn.option_templates_b[0], inc)
            for order, cmap, frame, famname in itertools.product(
                    range(2), range(2), ("enacted", "hypothetical"),
                    ("letters", "numbers")):
                item = _base_item(scn, inc, "F-P1-CONT", pair, cmap)
                item.format_id = "F-P1"
                item.display_order = order
                item.display_label_set = famname
                item.consequence_frame = frame
                item.option_text_by_sem = {"a": opt_a, "b": opt_b}
                first_sem, second_sem = ("a", "b") if order == 0 else ("b", "a")
                texts = {"a": opt_a, "b": opt_b}
                labels = {"letters": ("A", "B"), "numbers": ("1", "2")}[famname]
                codes_disp = (item.response_code_by_sem[first_sem],
                              item.response_code_by_sem[second_sem])
                prompt, spans = render_fp1(
                    framing=framing, frame_text=_FRAME_TEXT[frame],
                    labels=labels, codes_in_display_order=codes_disp,
                    options_in_display_order=(texts[first_sem],
                                              texts[second_sem]),
                    reply_codes_in_list_order=codes_disp,
                )
                item.user_prompt = prompt
                item.site_char_spans = spans
                item.valid_codes_in_display_order = codes_disp
                items.append(finalize_identity(item))
    return items


def _build_dev(families: CodebookFamilies) -> list[BankItem]:
    items = []
    for scn in DEV_SCENARIOS:
        for idx, inc in enumerate(scn.incidentals):
            pair = rotation_pair_for(families, "AR", scn.scenario_id, idx,
                                     inc.incidental_split)
            framing = scn.render(scn.framing_templates[0], inc)
            opt_a = scn.render(scn.option_templates_a[0], inc)
            opt_b = scn.render(scn.option_templates_b[0], inc)
            # F-SYM: 2 order x 2 cmap x 2 frame
            for order, cmap, frame in itertools.product(
                    range(2), range(2), ("enacted", "hypothetical")):
                item = _base_item(scn, inc, "B-DEV", pair, cmap)
                item.prompt_subset = "dev"
                items.append(_finish_ar_fsym(
                    item, scn, inc, display_order=order, paraphrase_id=0,
                    consequence_frame=frame, context_statement=None))
            # F-P1: 2 order x 2 cmap x 2 frame (letters)
            for order, cmap, frame in itertools.product(
                    range(2), range(2), ("enacted", "hypothetical")):
                item = _base_item(scn, inc, "B-DEV", pair, cmap)
                item.prompt_subset = "dev"
                item.format_id = "F-P1"
                item.display_order = order
                item.display_label_set = "letters"
                item.consequence_frame = frame
                item.option_text_by_sem = {"a": opt_a, "b": opt_b}
                first_sem, second_sem = ("a", "b") if order == 0 else ("b", "a")
                texts = {"a": opt_a, "b": opt_b}
                codes_disp = (item.response_code_by_sem[first_sem],
                              item.response_code_by_sem[second_sem])
                prompt, spans = render_fp1(
                    framing=framing, frame_text=_FRAME_TEXT[frame],
                    labels=("A", "B"), codes_in_display_order=codes_disp,
                    options_in_display_order=(texts[first_sem],
                                              texts[second_sem]),
                    reply_codes_in_list_order=codes_disp,
                )
                item.user_prompt = prompt
                item.site_char_spans = spans
                item.valid_codes_in_display_order = codes_disp
                items.append(finalize_identity(item))
            # F-COMMIT: 2 orders (enacted, cmap 0) — dev diagnostic only
            for order in range(2):
                item = _base_item(scn, inc, "B-DEV", pair, 0)
                item.prompt_subset = "dev"
                item.format_id = "F-COMMIT"
                item.display_order = order
                item.consequence_frame = "enacted"
                item.option_text_by_sem = {"a": opt_a, "b": opt_b}
                first_sem, second_sem = ("a", "b") if order == 0 else ("b", "a")
                texts = {"a": opt_a, "b": opt_b}
                prompt, spans = render_fcommit(
                    framing=framing, frame_text=ENACTED_FRAME_TEXT,
                    first_code=item.response_code_by_sem[first_sem],
                    first_text=texts[first_sem],
                    second_code=item.response_code_by_sem[second_sem],
                    second_text=texts[second_sem],
                )
                item.user_prompt = prompt
                item.site_char_spans = order_site_spans(
                    spans, display_order=order, channel="AR")
                item.valid_codes_in_display_order = (
                    item.response_code_by_sem[first_sem],
                    item.response_code_by_sem[second_sem],
                )
                items.append(finalize_identity(item))
    return items


def build_bank(families: CodebookFamilies) -> list[BankItem]:
    items = []
    items.extend(_build_dev(families))
    items.extend(_build_surf(families))
    items.extend(_build_arb3(families))
    items.extend(_build_mech(families))
    items.extend(_build_pcmech(families))
    items.extend(_build_canon(families))
    items.extend(_build_pc(families))
    items.extend(_build_nc(families))
    items.extend(_build_ro(families))
    items.extend(_build_continuity(families))
    return items


def bank_content_hash(items: list[BankItem]) -> str:
    return canonical_hash([it.scientific_content_hash for it in items])


# Audits -------------------------------------------------------------------

def audit_bank(items: list[BankItem], families: CodebookFamilies) -> dict[str, Any]:
    failures: list[str] = []

    counts: dict[str, int] = {}
    for it in items:
        counts[it.bank] = counts.get(it.bank, 0) + 1
    for bank, expected in EXPECTED_COUNTS.items():
        got = counts.get(bank, 0)
        if got != expected:
            failures.append(f"count {bank}: got {got}, expected {expected}")
    if len(items) != EXPECTED_TOTAL:
        failures.append(f"total {len(items)} != {EXPECTED_TOTAL}")

    ids = [it.item_id for it in items]
    if len(set(ids)) != len(ids):
        failures.append("duplicate item_ids")
    hashes = [it.scientific_content_hash for it in items]
    if len(set(hashes)) != len(hashes):
        failures.append("duplicate scientific hashes")

    ar_codes = set(families.all_codes("AR"))
    ro_codes = set(families.all_codes("RO"))
    if ar_codes & ro_codes:
        failures.append("AR/RO alphabets overlap")
    for it in items:
        pool = ro_codes if it.channel == "RO" else ar_codes
        if not set(it.response_code_by_sem.values()) <= pool:
            failures.append(f"{it.item_id}: code outside channel alphabet")
            break

    # reserved family never in train/validation (E5)
    for it in items:
        if it.codebook_reserved and it.incidental_split != "holdout":
            failures.append(f"reserved pair on {it.incidental_split}: {it.item_id}")
            break

    # factor balance within scenario x bank x channel over non-reserved rows
    from collections import Counter, defaultdict
    cells: dict[tuple, Counter] = defaultdict(Counter)
    for it in items:
        if it.codebook_reserved:
            continue
        key = (it.bank, it.scenario_id, it.channel, it.format_id)
        cells[key][(it.display_order, it.code_map_index,
                    str(it.consequence_frame), it.paraphrase_id,
                    it.context_strength, str(it.canon_context),
                    str(it.display_label_set), str(it.label_assignment),
                    str(it.reply_list_order), str(it.pcmech_difficulty))] += 1
    for key, counter in cells.items():
        if len(set(counter.values())) != 1:
            failures.append(f"unbalanced factor grid in {key}: {dict(counter)}")

    # context strength independent of surfaces (schema §54.1): within every
    # scenario, each strength must appear equally often at each order/cmap
    strengths: dict[tuple, Counter] = defaultdict(Counter)
    for it in items:
        if it.bank in ("B-MECH", "B-PC-MECH") and not it.codebook_reserved:
            strengths[(it.scenario_id, it.display_order,
                       it.code_map_index)][it.context_strength] += 1
    for key, counter in strengths.items():
        if len(set(counter.values())) != 1:
            failures.append(f"strength/surface imbalance {key}")

    # AR<->RO pairing: every ARB3 (order, cmap) cell has exactly 2 RO twins
    ro_by_pair: dict[str, int] = Counter()
    for it in items:
        if it.channel == "RO" and not it.codebook_reserved:
            ro_by_pair[it.pair_key] += 1
    for it in items:
        if it.bank == "B-ARB3":
            n = ro_by_pair.get(it.pair_key, 0)
            if n != 2:
                failures.append(f"pair_key {it.pair_key}: {n} RO twins != 2")
                break

    # RO never binds; AR frozen banks bind (E7)
    for it in items:
        if it.channel == "RO" and it.binding_kind is not None:
            failures.append(f"RO row with binding: {it.item_id}")
            break
        if (it.bank in ("B-ARB3", "B-MECH", "B-PC-MECH", "B-PC")
                and it.channel == "AR" and it.binding_kind is None):
            failures.append(f"AR row missing binding: {it.item_id}")
            break
        if it.bank in ("B-MECH", "B-PC-MECH") and it.binding_kind == "model_microtask":
            failures.append(f"B-MECH microtask binding (E7): {it.item_id}")
            break

    # NC identity rules
    for it in items:
        if it.nc_family in ("nc_identical", "nc_code_only", "nc_context_null"):
            if it.option_text_by_sem["a"] != it.option_text_by_sem["b"]:
                failures.append(f"NC options differ: {it.item_id}")
                break

    # dev rows only in B-DEV; frozen banks never dev
    for it in items:
        if (it.bank == "B-DEV") != (it.prompt_subset == "dev"):
            failures.append(f"dev subset mismatch: {it.item_id}")
            break

    # no B-DEV text reused in a frozen bank (plan §9)
    dev_prompts = {it.user_prompt for it in items if it.bank == "B-DEV"}
    for it in items:
        if it.bank != "B-DEV" and it.user_prompt in dev_prompts:
            failures.append(f"frozen row duplicates dev prompt: {it.item_id}")
            break

    # F-SYM must have no display labels / repeated reply list (plan §11)
    for it in items:
        if it.format_id == "F-SYM":
            if it.display_label_set is not None:
                failures.append(f"F-SYM with labels: {it.item_id}")
                break
            if "Option A" in it.user_prompt or "Option 1" in it.user_prompt:
                failures.append(f"F-SYM label text: {it.item_id}")
                break
            body = it.user_prompt
            for code in it.valid_codes_in_display_order:
                if body.count(code) != 1:
                    failures.append(f"F-SYM code repeated: {it.item_id}")
                    break

    # sentinel constancy at char level (E6 token-level check at port audit)
    for it in items:
        if it.channel == "RO":
            if "Survey context complete." not in it.user_prompt:
                failures.append(f"RO sentinel missing: {it.item_id}")
                break
        elif it.format_id == "F-SYM":
            if "Context complete." not in it.user_prompt:
                failures.append(f"AR sentinel missing: {it.item_id}")
                break

    # split counts
    split_expect = {"B-ARB3": (12, 6, 6), "B-MECH": (16, 8, 8),
                    "B-PC-MECH": (16, 8, 8)}
    for bank, (tr, va, ho) in split_expect.items():
        for scn_id in {it.scenario_id for it in items if it.bank == bank}:
            incs = {(it.incidental_id, it.incidental_split)
                    for it in items if it.bank == bank
                    and it.scenario_id == scn_id}
            got = Counter(s for _, s in incs)
            if (got.get("train", 0), got.get("validation", 0),
                    got.get("holdout", 0)) != (tr, va, ho):
                failures.append(f"split counts {bank}/{scn_id}: {dict(got)}")

    return {
        "bank_version": BANK_VERSION,
        "counts": counts,
        "expected_counts": EXPECTED_COUNTS,
        "total": len(items),
        "n_failures": len(failures),
        "failures": failures[:40],
        "passed": not failures,
    }
