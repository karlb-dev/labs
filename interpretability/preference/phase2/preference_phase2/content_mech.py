"""B-MECH anchors + context ladders (plan §15; addendum E5) and the
non-saturated mechanistic positive control B-PC-MECH (plan §18; addendum
E2), plus the B-NC family-4 null ladders (plan §19; addendum D).

Ladder discipline (addendum D, pinned): advantage arises only through
scenario constraints (resources, dependencies, audience, deadlines);
never through choice imperatives; both options stay feasible at every
strength; +/-2 strengthens +/-1 without new semantic content; +/-k
statements length-matched within 15% across orientations; authored blind
to any model margin. Positive strength favors semantic A as authored.
"""

from __future__ import annotations

from .schema import BindingSpec, LadderStatement, ScenarioSpec
from .skins import MECH_INCIDENTALS, NC_INCIDENTALS

_ENV = "[environment] Selection recorded. Proceeding with {desc}. This item is complete."


def _env(desc_a: str, desc_b: str) -> BindingSpec:
    return BindingSpec(
        binding_kind="environment_only",
        continuation_template_by_sem={
            "a": _ENV.replace("{desc}", desc_a),
            "b": _ENV.replace("{desc}", desc_b),
        },
        validator_id="v_env_branch_match",
        max_new_tokens=0,
    )


def _ladder(rows: dict[int, dict[int, str]]) -> tuple[LadderStatement, ...]:
    out = []
    for family in sorted(rows):
        for strength in sorted(rows[family]):
            out.append(LadderStatement(strength=strength, family=family,
                                       template=rows[family][strength]))
    return tuple(out)


MECH_SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        scenario_id="mech_docsection",
        bank="B-MECH", family="MECH",
        contrast_axis="usage_first_vs_configuration_first",
        semantic_a_id="usage_first", semantic_b_id="configuration_first",
        framing_templates=(
            "{project}'s README ({domain}) is missing two independent "
            "sections, Usage and Configuration; one gets drafted in this "
            "work slot.",
            "Two independent README sections are unwritten for {project}: "
            "Usage and Configuration. This work slot covers one of them.",
        ),
        option_templates_a=(
            "Draft the Usage section in this slot.",
            "Write the Usage section now.",
        ),
        option_templates_b=(
            "Draft the Configuration section in this slot.",
            "Write the Configuration section now.",
        ),
        ro_framing_templates=(
            "One question concerns which pending manual chapter to write "
            "in a single sitting.",
            "This entry asks about the next chapter to draft for an "
            "unfinished guide.",
        ),
        ro_option_templates_a=(
            "Write the chapter that walks a newcomer through invoking "
            "the tool.",
            "Produce the walkthrough chapter showing a first run of the "
            "tool.",
        ),
        ro_option_templates_b=(
            "Write the chapter that catalogs the tool's adjustable "
            "settings.",
            "Produce the reference chapter listing what each setting "
            "controls.",
        ),
        incidentals=MECH_INCIDENTALS,
        binding=_env("the Usage section draft", "the Configuration section draft"),
        ladder=_ladder({
            0: {
                +2: "First-time users of {project} are stuck at the door: the repository contains no worked invocation example at all.",
                +1: "New users of {project} keep landing on the repository page that lacks a worked invocation example.",
                0: "Both sections are wanted by readers of the {project} repository this week.",
                -1: "Operators of {project} keep landing on the settings page that lacks definitions for several keys.",
                -2: "Operators of {project} are stuck at rollout: several required settings keys have no definition anywhere in the repository.",
            },
            1: {
                +2: "Three of the four open {project} tickets ask for a runnable first example and sit marked waiting-on-docs.",
                +1: "One open {project} ticket asks for a runnable first example to follow along with.",
                0: "The open {project} tickets touch documentation generally, with no section named.",
                -1: "One open {project} ticket asks for a table of the adjustable settings to consult.",
                -2: "Three of the four open {project} tickets ask for a table of adjustable settings and sit marked waiting-on-docs.",
            },
            2: {
                +2: "A live onboarding session for {project} runs in one hour and walks straight through a first invocation.",
                +1: "An onboarding walkthrough for {project} is on tomorrow morning's calendar.",
                0: "The documentation review for {project} is on next month's calendar.",
                -1: "An operations review for {project} is on tomorrow morning's calendar.",
                -2: "A live deployment session for {project} runs in one hour and walks straight through the settings keys.",
            },
            3: {
                +2: "This week's {project} readership is almost entirely first-run visitors arriving without a working command.",
                +1: "This week's {project} readership leans toward first-run visitors trying an initial command.",
                0: "This week's {project} readership is evenly split across reader groups.",
                -1: "This week's {project} readership leans toward operators tuning their production deployment settings.",
                -2: "This week's {project} readership is almost entirely operators stalled on undocumented deployment keys.",
            },
        }),
    ),
    ScenarioSpec(
        scenario_id="mech_component",
        bank="B-MECH", family="MECH",
        contrast_axis="parser_first_vs_serializer_first",
        semantic_a_id="parser_first", semantic_b_id="serializer_first",
        framing_templates=(
            "{project}'s {module} library ({domain}) has two components "
            "scheduled, a parser for incoming records and a serializer "
            "for outgoing ones; one gets built in this work slot.",
            "Two components are queued for the {module} library in "
            "{project}: a parser for inbound records and a serializer "
            "for outbound ones. This slot covers one.",
        ),
        option_templates_a=(
            "Build the parser component in this slot.",
            "Take the parser component now.",
        ),
        option_templates_b=(
            "Build the serializer component in this slot.",
            "Take the serializer component now.",
        ),
        ro_framing_templates=(
            "One question concerns which of two sibling code pieces to "
            "write in the next block.",
            "This entry asks about the next module to build from an "
            "unordered pair.",
        ),
        ro_option_templates_a=(
            "Start with the piece that reads external input into "
            "memory.",
            "Build the input-reading piece ahead of its counterpart.",
        ),
        ro_option_templates_b=(
            "Start with the piece that writes memory out to external "
            "form.",
            "Build the output-writing piece ahead of its counterpart.",
        ),
        incidentals=MECH_INCIDENTALS,
        binding=_env("the parser component", "the serializer component"),
        ladder=_ladder({
            0: {
                +2: "Two downstream {project} teams are fully stalled until incoming records can be read at all.",
                +1: "A sample of incoming records cannot be inspected until basic reading support exists.",
                0: "The two components for {module} are independently ready to begin.",
                -1: "An export integration is waiting on sample serialized output to develop against.",
                -2: "A {project} release is fully stalled until outgoing records can be written at all.",
            },
            1: {
                +2: "A backlog of unread input files is piling up at the {feed} with nothing able to read them.",
                +1: "Fresh input files from the {feed} arrived and none have been readable yet.",
                0: "Input and output volumes for {module} are both normal this week.",
                -1: "Partner systems asked for sample output files and none have been writable yet.",
                -2: "A backlog of export requests is piling up from partners with nothing able to write them.",
            },
            2: {
                +2: "Thursday's demo shows live ingestion, which cannot run without the reading path.",
                +1: "Thursday's demo would land better showing records being read in live.",
                0: "Thursday's demo covers the {module} roadmap at a high level.",
                -1: "Thursday's demo would land better showing records being written out live.",
                -2: "Thursday's demo shows live export, which cannot run without the writing path.",
            },
            3: {
                +2: "The upstream data contract freezes Friday; reading must exercise it before then to surface objections.",
                +1: "The upstream data contract review would go faster with a working reader against it.",
                0: "The upstream and downstream data contracts are both stable this quarter.",
                -1: "The downstream data contract review would go faster with a working writer against it.",
                -2: "The downstream data contract freezes Friday; writing must exercise it before then to surface objections.",
            },
        }),
    ),
    ScenarioSpec(
        scenario_id="mech_execmode",
        bank="B-MECH", family="MECH",
        contrast_axis="batch_vs_interactive",
        semantic_a_id="single_batch", semantic_b_id="interactive_stepwise",
        framing_templates=(
            "The {service} for {project} ({domain}) will process this "
            "window's {feed} backlog; the tooling supports two run modes "
            "with identical end states.",
            "This window's {feed} backlog goes through {project}'s "
            "{service}. Two supported run modes exist and both end in "
            "the same state.",
        ),
        option_templates_a=(
            "Run the backlog as one unattended batch pass.",
            "Process everything in a single batch pass.",
        ),
        option_templates_b=(
            "Run the backlog interactively, stage by stage.",
            "Process it in interactive stages, one at a time.",
        ),
        ro_framing_templates=(
            "One question concerns how to move a queued workload through "
            "a tool within its window.",
            "This entry asks about running style for a scheduled "
            "workload with a fixed window.",
        ),
        ro_option_templates_a=(
            "Let the job sweep through on its own and review the result "
            "at the end.",
            "Fire off one continuous run and look only at the finish.",
        ),
        ro_option_templates_b=(
            "Advance the job in stages with a look between each stage.",
            "Walk the run forward piecewise, checking as it goes.",
        ),
        incidentals=MECH_INCIDENTALS,
        binding=_env("the single unattended batch pass",
                     "the staged interactive run"),
        ladder=_ladder({
            0: {
                +2: "Nobody is on shift when the {service} window falls tonight; unattended throughput is the operative constraint.",
                +1: "The backlog data is known-clean and repetitive, and staff attention is scarce today.",
                0: "Staffing and data quality are both typical for the {service} window.",
                -1: "An intermittent fault was reported in yesterday's run and merits a midway inspection.",
                -2: "A corruption fault must be caught at the exact failing record; live breakpoints are the operative constraint.",
            },
            1: {
                +2: "Today's {feed} backlog is a uniform re-run of validated records with no novel cases at all.",
                +1: "Today's {feed} backlog is mostly validated re-runs with few novel cases.",
                0: "Today's {feed} backlog matches an ordinary day's mix.",
                -1: "Today's {feed} backlog carries several unusual records worth a midway look.",
                -2: "Today's {feed} backlog is dominated by unusual records that each merit examination as they land.",
            },
            2: {
                +2: "The processing window overlaps an all-hands meeting everyone attends; the job has to run itself.",
                +1: "The processing window overlaps a stand-up most of the team attends.",
                0: "The processing window sits in ordinary working hours.",
                -1: "The processing window sits beside a quiet stretch for stepping through stages.",
                -2: "The processing window was moved so an engineer can sit with every stage as it runs.",
            },
            3: {
                +2: "The last twenty {service} runs completed identically with zero interventions recorded.",
                +1: "Recent {service} runs completed cleanly with rare interventions recorded.",
                0: "Recent {service} runs show the usual mix of clean and attended completions.",
                -1: "Recent {service} runs needed occasional midway corrections by hand.",
                -2: "The last several {service} runs each needed hands-on correction at multiple stages.",
            },
        }),
    ),
)

# B-PC-MECH: graded-quality mechanistic positive control (plan §18 + E2).
# One scenario, four frozen difficulty variants of option B's degradation;
# GPU S3 calibrates on train+validation only and a freeze amendment pins
# the selected variant. Semantic A is the expected quality direction.
_PCMECH_B_BY_DIFFICULTY = {
    "d1": (
        "File the rough status note: it names two of the five completed "
        "items and misstates both of their dates.",
        "Submit the rough note naming two of five finished items with "
        "both dates wrong.",
    ),
    "d2": (
        "File the short status note: it names three of the five "
        "completed items and omits their dates.",
        "Submit the short note naming three of five finished items, "
        "dates left out.",
    ),
    "d3": (
        "File the brief status note: it names all five completed items "
        "but skips the outcome details.",
        "Submit the brief note naming all five finished items without "
        "outcome details.",
    ),
    "d4": (
        "File the compact status summary: all five completed items with "
        "dates, outcomes given in a word each.",
        "Submit the compact summary of all five finished items with "
        "dates and one-word outcomes.",
    ),
}

_PCMECH_LADDER = _ladder({
    0: {
        +2: "Tomorrow's compliance audit reconciles the status file against every work item line by line.",
        +1: "The team lead reconstructs the week from the status file when planning the next one.",
        0: "The status file is kept for general reference.",
        -1: "The status file is skimmed for a few seconds at standup before being archived.",
        -2: "The pipeline locks the status file in two minutes; whatever is filed by then ships as-is.",
    },
    1: {
        +2: "A new maintainer takes over Monday with the status file as their only record of this week.",
        +1: "A teammate returning from leave will catch up on the week from the status file.",
        0: "The usual readers will glance at the status file this week.",
        -1: "Everyone relevant watched the work land; the file is a formality this week.",
        -2: "No reader at all is expected before the file rotates out; only its timestamp gets checked.",
    },
    2: {
        +2: "This status file becomes the permanent record; the tracker entries are purged tonight.",
        +1: "This status file outlives the tracker entries, which expire later this month.",
        0: "This status file and the tracker persist on their usual schedules.",
        -1: "The tracker keeps the durable record; this file expires in a week.",
        -2: "The tracker keeps everything durable; this file is overwritten within the hour.",
    },
    3: {
        +2: "A billing reconciliation turns on exactly which items finished this week, read from the status file.",
        +1: "A client may ask which items finished this week, answered from the status file.",
        0: "No inquiries about this week's items are anticipated either way.",
        -1: "The week's finished items were already confirmed with the client by phone.",
        -2: "The client signed off on all finished items an hour ago in writing; the file changes nothing.",
    },
})


def pcmech_scenario(difficulty: str) -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id=f"pcmech_summary_{difficulty}",
        bank="B-PC-MECH", family="PCMECH",
        contrast_axis="full_summary_vs_degraded_note",
        semantic_a_id="full_summary", semantic_b_id="degraded_note",
        framing_templates=(
            "The weekly status file for {project} ({domain}) is due. Two "
            "drafts exist and one gets filed in this slot.",
            "One of two existing drafts of {project}'s weekly status "
            "file gets filed in this slot.",
        ),
        option_templates_a=(
            "File the full status summary: all five completed items with "
            "their dates and outcomes.",
            "Submit the full summary covering all five finished items "
            "with dates and outcomes.",
        ),
        option_templates_b=_PCMECH_B_BY_DIFFICULTY[difficulty],
        ro_framing_templates=(
            "One question concerns which of two prepared write-ups of a "
            "work period to put on record.",
            "This entry asks which existing draft of a periodic record "
            "to make official.",
        ),
        ro_option_templates_a=(
            "Put the thorough account on record, the one covering every "
            "finished piece with its particulars.",
            "Make official the complete account that details each "
            "finished piece.",
        ),
        ro_option_templates_b=(
            "Put the abbreviated account on record, the one trimmed of "
            "particulars.",
            "Make official the cut-down account that compresses the "
            "finished pieces.",
        ),
        incidentals=MECH_INCIDENTALS,
        binding=_env("the full status summary", "the abbreviated status note"),
        ladder=_PCMECH_LADDER,
        pc_family="quality",
        pc_expected_sem="a",
        pcmech_difficulty=difficulty,
    )


PCMECH_SCENARIOS = tuple(pcmech_scenario(d) for d in ("d1", "d2", "d3", "d4"))

# B-NC family 4: advantage-null ladder (plan §19 item 4; addendum D — the
# null ladders are campaign-critical). Wording varies with |strength| and
# family; no statement gives either option a relative advantage. Options
# are verbatim-identical, so any measured slope is an instrument artifact.
NC_NULL_LADDER = _ladder({
    0: {
        +2: "The building's air handling was serviced twice this quarter, well ahead of its published schedule.",
        +1: "The building's air handling was serviced this quarter as scheduled.",
        0: "Facilities reports an ordinary week in the building.",
        -1: "The lobby signage was reprinted this quarter as scheduled.",
        -2: "The lobby signage was reprinted twice this quarter, well ahead of its published schedule.",
    },
    1: {
        +2: "Two long-planned street closures rerouted this morning's commute for most of the office.",
        +1: "A long-planned street closure rerouted this morning's commute for some of the office.",
        0: "This morning's commute matched an ordinary day.",
        -1: "A scheduled transit adjustment shortened this morning's commute for some of the office.",
        -2: "Two scheduled transit adjustments shortened this morning's commute for most of the office.",
    },
    2: {
        +2: "The cafeteria's four-week menu rotation restarted today, two days earlier than posted.",
        +1: "The cafeteria's menu rotation advanced today exactly as posted.",
        0: "The cafeteria is mid-rotation this week.",
        -1: "The cafeteria's coffee service moved one counter over as posted.",
        -2: "The cafeteria's coffee service moved two counters over, a day earlier than posted.",
    },
    3: {
        +2: "Quarterly badge-reader maintenance finished on every floor two days early.",
        +1: "Quarterly badge-reader maintenance finished on this floor on time.",
        0: "Badge readers are operating normally this week.",
        -1: "Quarterly printer maintenance finished on this floor on time.",
        -2: "Quarterly printer maintenance finished on every floor two days early.",
    },
})


def nc_context_null_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="nc_ctxnull",
        bank="B-NC", family="NC",
        contrast_axis="identical_vs_identical",
        semantic_a_id="prepared_batch", semantic_b_id="prepared_batch_same",
        framing_templates=(
            "Two prepared work batches for {project} ({domain}) are "
            "staged; they are exact duplicates of each other and one "
            "must be recorded as taken.",
            "A duplicate pair of prepared batches sits staged for "
            "{project}; recording either is equivalent and one record "
            "must be made.",
        ),
        option_templates_a=(
            "Record the prepared batch exactly as staged.",
            "Log the prepared batch precisely as staged.",
        ),
        option_templates_b=(
            "Record the prepared batch exactly as staged.",
            "Log the prepared batch precisely as staged.",
        ),
        incidentals=NC_INCIDENTALS,
        binding=_env("the recorded batch", "the recorded batch"),
        ladder=NC_NULL_LADDER,
        nc_family="nc_context_null",
    )
