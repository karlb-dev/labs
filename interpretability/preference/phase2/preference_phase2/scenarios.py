"""Scenario rosters + authoring self-check (plan Part IV)."""

from __future__ import annotations

from .content_arb3 import ARB3_SCENARIOS
from .content_aux import (CANON_SCENARIOS, DEV_SCENARIOS, NC_SCENARIOS,
                          PC_SCENARIOS, SURF_SCENARIOS, canon_contexts)
from .content_mech import (MECH_SCENARIOS, NC_NULL_LADDER, PCMECH_SCENARIOS,
                           nc_context_null_scenario)
from .schema import ScenarioSpec

NC_CTXNULL = nc_context_null_scenario()

ALL_NC_SCENARIOS = (*NC_SCENARIOS, NC_CTXNULL)

FROZEN_SCENARIOS: tuple[ScenarioSpec, ...] = (
    *ARB3_SCENARIOS, *MECH_SCENARIOS, *PCMECH_SCENARIOS, *CANON_SCENARIOS,
    *PC_SCENARIOS, *ALL_NC_SCENARIOS, *SURF_SCENARIOS,
)
ALL_SCENARIOS: tuple[ScenarioSpec, ...] = (*FROZEN_SCENARIOS, *DEV_SCENARIOS)

# Ladder-line wordlist guard (addendum D): scenario constraints only,
# never choice imperatives.
LADDER_FORBIDDEN_WORDS = ("choose", "prefer", "pick", "select", "you should")


def scenario_by_id(scenario_id: str) -> ScenarioSpec:
    for s in ALL_SCENARIOS:
        if s.scenario_id == scenario_id:
            return s
    raise KeyError(scenario_id)


def self_check() -> None:
    ids = [s.scenario_id for s in ALL_SCENARIOS]
    assert len(ids) == len(set(ids)), "duplicate scenario ids"
    assert len(ARB3_SCENARIOS) == 12
    assert len(MECH_SCENARIOS) == 3
    assert len(PCMECH_SCENARIOS) == 4
    assert len(CANON_SCENARIOS) == 6
    assert len(PC_SCENARIOS) == 6
    assert len(ALL_NC_SCENARIOS) == 6
    assert len(SURF_SCENARIOS) == 4

    micro = [s for s in ARB3_SCENARIOS
             if s.binding and s.binding.binding_kind == "model_microtask"]
    assert len(micro) == 6, "plan §14: at least six microtask scenarios"

    canon_roles = [s.canon_role for s in CANON_SCENARIOS]
    assert canon_roles.count("discovery") == 3
    assert canon_roles.count("heldout") == 3

    for s in ALL_SCENARIOS:
        for inc in s.incidentals:
            for tpl in (*s.framing_templates, *s.option_templates_a,
                        *s.option_templates_b, *s.ro_framing_templates,
                        *s.ro_option_templates_a, *s.ro_option_templates_b):
                rendered = s.render(tpl, inc)
                assert "{" not in rendered and "}" not in rendered, (
                    f"unfilled placeholder in {s.scenario_id}: {rendered[:60]}")
            if s.binding:
                for tpl in s.binding.continuation_template_by_sem.values():
                    rendered = s.render(tpl, inc)
                    assert "{" not in rendered.replace("{}", ""), (
                        f"unfilled binding in {s.scenario_id}")
        # NC identical families must be verbatim-identical per paraphrase
        if s.nc_family in ("nc_identical", "nc_code_only", "nc_context_null"):
            assert s.option_templates_a == s.option_templates_b, s.scenario_id
        if s.family in ("ARB",):
            assert s.option_templates_a != s.option_templates_b, s.scenario_id
            assert len(s.framing_templates) == 2, s.scenario_id
            assert len(s.ro_framing_templates) == 2, s.scenario_id
        if s.family == "PC":
            assert s.pc_expected_sem == "a" and s.pc_family, s.scenario_id
        if s.bank in ("B-ARB3", "B-MECH", "B-PC-MECH"):
            assert s.binding is not None, s.scenario_id
        # ladder guard: forbidden imperative words never appear in ladder
        # lines (case-insensitive), and each family covers all 5 strengths
        if s.ladder:
            fams = {}
            for st in s.ladder:
                fams.setdefault(st.family, set()).add(st.strength)
                low = st.template.lower()
                for w in LADDER_FORBIDDEN_WORDS:
                    assert w not in low, (
                        f"ladder guard: {w!r} in {s.scenario_id} "
                        f"f{st.family}s{st.strength}")
            assert all(v == {-2, -1, 0, 1, 2} for v in fams.values()), (
                f"incomplete ladder {s.scenario_id}")
            assert set(fams) == {0, 1, 2, 3}, f"ladder families {s.scenario_id}"
    # canon contexts exist for all six axes
    for s in CANON_SCENARIOS:
        ctx = canon_contexts(s)
        assert set(ctx) == {"neutral", "favor_a", "favor_b"}
        for inc in s.incidentals:
            for t in ctx.values():
                rendered = s.render(t, inc)
                assert "{" not in rendered, s.scenario_id
