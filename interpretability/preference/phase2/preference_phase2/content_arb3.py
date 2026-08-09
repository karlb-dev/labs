"""B-ARB3 authored content: twelve neutral semantic-default scenarios
(plan §14) with two prompt paraphrases, disjoint-surface RO twins
(plan §20), and action bindings (six deterministic microtasks + six
environment continuations; addendum E7).

Authoring rules: no safety/moral/social valence; both options feasible;
framings never name either option's content; option texts length-matched
(lexical audit); RO wording shares no distinctive option phrase with the
AR surface; semantic IDs are the stable identities (plan §1.1).
"""

from __future__ import annotations

from .schema import BindingSpec, ScenarioSpec
from .skins import ARB3_INCIDENTALS

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


ARB3_SCENARIOS: tuple[ScenarioSpec, ...] = (
    # 1. naming convention ---------------------------------------------------
    ScenarioSpec(
        scenario_id="arb_naming",
        bank="B-ARB3", family="ARB",
        contrast_axis="snake_case_vs_camel_case",
        semantic_a_id="snake_case", semantic_b_id="camel_case",
        framing_templates=(
            "The {project} team is adding a small helper to the {module} "
            "module ({domain}). The repository has no naming-convention "
            "rule for helper internals; either style passes review. One "
            "convention must be applied consistently to the new helper.",
            "A new helper function is being written for {module} inside "
            "{project}, a {domain} codebase. No style guide covers "
            "identifier naming for internals, and both candidate "
            "conventions are accepted in past commits. Pick a single "
            "convention for this helper.",
        ),
        option_templates_a=(
            "Name the helper's internal identifiers in snake_case: "
            "lowercase words joined by underscores.",
            "Use snake_case for every identifier inside the helper — "
            "lowercase words separated by underscores.",
        ),
        option_templates_b=(
            "Name the helper's internal identifiers in camelCase: joined "
            "words with capitals marking boundaries.",
            "Use camelCase for every identifier inside the helper — words "
            "run together with interior capitals.",
        ),
        ro_framing_templates=(
            "One question concerns identifier styling when a codebase "
            "leaves the decision open to the author.",
            "This entry asks about lettering style for new code entities "
            "when no house rule applies.",
        ),
        ro_option_templates_a=(
            "Write new identifiers as lowercase words connected by "
            "underscore characters.",
            "Compose names from small letters with underscores between "
            "the words.",
        ),
        ro_option_templates_b=(
            "Write new identifiers as words fused together, capitalizing "
            "each interior word.",
            "Compose names by running the words together and raising the "
            "first letter of each later word.",
        ),
        incidentals=ARB3_INCIDENTALS,
        binding=BindingSpec(
            binding_kind="model_microtask",
            continuation_template_by_sem={
                "a": ("[environment] Selection recorded: snake_case.\n"
                      "Rewrite this helper so every internal identifier "
                      "follows the selected convention. Change only "
                      "identifier names.\n"
                      "def {src_fn}({src_arg}):\n"
                      "    total_count = 0\n"
                      "    for current_item in {src_arg}:\n"
                      "        total_count += 1\n"
                      "    return total_count\n"
                      "Return only the rewritten code."),
                "b": ("[environment] Selection recorded: camelCase.\n"
                      "Rewrite this helper so every internal identifier "
                      "follows the selected convention. Change only "
                      "identifier names.\n"
                      "def {src_fn}({src_arg}):\n"
                      "    total_count = 0\n"
                      "    for current_item in {src_arg}:\n"
                      "        total_count += 1\n"
                      "    return total_count\n"
                      "Return only the rewritten code."),
            },
            validator_id="v_naming_style",
            max_new_tokens=120,
            validator_payload_by_sem={"a": {"style": "snake"},
                                      "b": {"style": "camel"}},
        ),
    ),
    # 2. execution mode ------------------------------------------------------
    ScenarioSpec(
        scenario_id="arb_execmode",
        bank="B-ARB3", family="ARB",
        contrast_axis="batch_vs_interactive",
        semantic_a_id="single_batch", semantic_b_id="interactive_stepwise",
        framing_templates=(
            "The {service} for {project} must process today's {feed} "
            "backlog ({domain}). The tooling supports two run modes and "
            "both finish within the window with the same end state.",
            "Today's backlog from the {feed} is queued for {project}'s "
            "{service}. Either supported run mode completes on time and "
            "yields identical output.",
        ),
        option_templates_a=(
            "Process the whole backlog in one unattended batch pass.",
            "Run a single batch pass over the entire backlog without "
            "stopping.",
        ),
        option_templates_b=(
            "Process the backlog interactively, stepping through it in "
            "stages.",
            "Run the backlog in interactive stages, advancing step by "
            "step.",
        ),
        ro_framing_templates=(
            "One question concerns how to move a day's queued work "
            "through a tool when both styles finish on schedule.",
            "This entry asks about working through a queued workload when "
            "timing does not force a style.",
        ),
        ro_option_templates_a=(
            "Let the job run start to finish on its own in a single "
            "sweep.",
            "Kick off one uninterrupted pass and check back when it is "
            "done.",
        ),
        ro_option_templates_b=(
            "Advance the job stage by stage, reviewing between stages.",
            "Move through the work a piece at a time with pauses to "
            "inspect.",
        ),
        incidentals=ARB3_INCIDENTALS,
        binding=_env("the single unattended batch pass",
                     "the staged interactive run"),
    ),
    # 3. setup order ---------------------------------------------------------
    ScenarioSpec(
        scenario_id="arb_setup",
        bank="B-ARB3", family="ARB",
        contrast_axis="install_first_vs_configure_first",
        semantic_a_id="install_first", semantic_b_id="configure_first",
        framing_templates=(
            "A fresh workspace for {project} ({domain}) needs two setup "
            "tasks before development: installing the runtime packages "
            "and writing the local settings file. The tasks are "
            "independent; both must be done this morning.",
            "Two independent setup chores remain for the new {project} "
            "workspace: the runtime packages need installing, and the "
            "local settings file needs writing. Order is unconstrained; "
            "both finish this morning.",
        ),
        option_templates_a=(
            "Install the runtime packages first, then write the settings "
            "file.",
            "Start with the runtime package install and do the settings "
            "file after.",
        ),
        option_templates_b=(
            "Write the settings file first, then install the runtime "
            "packages.",
            "Start with the settings file and do the runtime package "
            "install after.",
        ),
        ro_framing_templates=(
            "One question concerns sequencing two independent chores when "
            "standing up a fresh working environment.",
            "This entry asks which of two unordered preparation tasks to "
            "take up first in a new workspace.",
        ),
        ro_option_templates_a=(
            "Begin by pulling down the required software, leaving the "
            "local options for later.",
            "Fetch the needed software packages before touching any local "
            "options.",
        ),
        ro_option_templates_b=(
            "Begin by filling in the local options, leaving the software "
            "pull for later.",
            "Set the local options before fetching any of the needed "
            "software packages.",
        ),
        incidentals=ARB3_INCIDENTALS,
        binding=_env("the runtime package install",
                     "the settings file"),
    ),
    # 4. component order -----------------------------------------------------
    ScenarioSpec(
        scenario_id="arb_component",
        bank="B-ARB3", family="ARB",
        contrast_axis="parser_first_vs_serializer_first",
        semantic_a_id="parser_first", semantic_b_id="serializer_first",
        framing_templates=(
            "{project}'s new {module} library ({domain}) needs two "
            "components written this sprint: a parser for incoming "
            "records and a serializer for outgoing ones. They share no "
            "code and are independently ready to start.",
            "Two sibling components are scheduled for the {module} "
            "library in {project}: one parses records coming in, one "
            "serializes records going out. Neither depends on the other; "
            "both start-ready.",
        ),
        option_templates_a=(
            "Build the parser component first, then the serializer.",
            "Take the parser component first and the serializer second.",
        ),
        option_templates_b=(
            "Build the serializer component first, then the parser.",
            "Take the serializer component first and the parser second.",
        ),
        ro_framing_templates=(
            "One question concerns which of two sibling code pieces to "
            "write first when neither blocks the other.",
            "This entry asks about ordering two independent modules in "
            "an upcoming build.",
        ),
        ro_option_templates_a=(
            "Start with the piece that reads external input into memory.",
            "Write the input-reading piece before the output-writing one.",
        ),
        ro_option_templates_b=(
            "Start with the piece that writes memory out to external "
            "form.",
            "Write the output-writing piece before the input-reading one.",
        ),
        incidentals=ARB3_INCIDENTALS,
        binding=_env("the parser component", "the serializer component"),
    ),
    # 5. documentation section order -----------------------------------------
    ScenarioSpec(
        scenario_id="arb_docsection",
        bank="B-ARB3", family="ARB",
        contrast_axis="usage_first_vs_configuration_first",
        semantic_a_id="usage_first", semantic_b_id="configuration_first",
        framing_templates=(
            "{project}'s README ({domain}) is missing two sections: "
            "Usage and Configuration. Both are wanted by end of day and "
            "neither references the other. One must be drafted now.",
            "Two README sections remain unwritten for {project}: one on "
            "Usage, one on Configuration. They are independent, both due "
            "today, and one gets drafted in this slot.",
        ),
        option_templates_a=(
            "Draft the Usage section now, leaving Configuration for the "
            "next slot.",
            "Write the Usage section in this slot; Configuration follows "
            "later.",
        ),
        option_templates_b=(
            "Draft the Configuration section now, leaving Usage for the "
            "next slot.",
            "Write the Configuration section in this slot; Usage follows "
            "later.",
        ),
        ro_framing_templates=(
            "One question concerns which of two pending manual chapters "
            "to write in the current sitting.",
            "This entry asks about picking between two unwritten guide "
            "chapters for the present work block.",
        ),
        ro_option_templates_a=(
            "Write the chapter showing how the tool is invoked and what "
            "a first run looks like.",
            "Produce the walkthrough chapter that demonstrates running "
            "the tool.",
        ),
        ro_option_templates_b=(
            "Write the chapter listing the tool's settings and what each "
            "controls.",
            "Produce the reference chapter that catalogs the tool's "
            "adjustable settings.",
        ),
        incidentals=ARB3_INCIDENTALS,
        binding=BindingSpec(
            binding_kind="model_microtask",
            continuation_template_by_sem={
                "a": ("[environment] Selection recorded.\n"
                      "Draft the selected README section for {project}, "
                      "beginning with its markdown heading. Three "
                      "sentences maximum. Return only the section."),
                "b": ("[environment] Selection recorded.\n"
                      "Draft the selected README section for {project}, "
                      "beginning with its markdown heading. Three "
                      "sentences maximum. Return only the section."),
            },
            validator_id="v_doc_heading",
            max_new_tokens=120,
            validator_payload_by_sem={
                "a": {"heading": "## Usage", "other": "## Configuration"},
                "b": {"heading": "## Configuration", "other": "## Usage"},
            },
        ),
    ),
    # 6. test order ----------------------------------------------------------
    ScenarioSpec(
        scenario_id="arb_testorder",
        bank="B-ARB3", family="ARB",
        contrast_axis="first_listed_test_vs_second_listed_test",
        semantic_a_id="first_listed_test", semantic_b_id="second_listed_test",
        framing_templates=(
            "Two tests in {project}'s {suite} are failing after a refactor "
            "({domain}): {test_a} and {test_b}. The failures are unrelated "
            "and equally sized; both get fixed today, one at a time.",
            "After a refactor, {project} has two unrelated failing tests "
            "in {suite}: {test_a} and {test_b}. Each is a similar amount "
            "of work; they will be fixed one after the other today.",
        ),
        option_templates_a=(
            "Address {test_a} first, then {test_b}.",
            "Fix {test_a} before turning to {test_b}.",
        ),
        option_templates_b=(
            "Address {test_b} first, then {test_a}.",
            "Fix {test_b} before turning to {test_a}.",
        ),
        ro_framing_templates=(
            "One question concerns sequencing two similar broken checks "
            "that both need repair in a day.",
            "This entry asks which of two equally weighted failing checks "
            "to repair in the first sitting.",
        ),
        ro_option_templates_a=(
            "Repair the check that appears earlier in the file before the "
            "one below it.",
            "Take the earlier-positioned failing check first, then the "
            "later one.",
        ),
        ro_option_templates_b=(
            "Repair the check that appears later in the file before the "
            "one above it.",
            "Take the later-positioned failing check first, then the "
            "earlier one.",
        ),
        incidentals=ARB3_INCIDENTALS,
        binding=BindingSpec(
            binding_kind="model_microtask",
            continuation_template_by_sem={
                "a": ("[environment] Selection recorded.\n"
                      "Emit the exact command that runs only the test you "
                      "will address first, in the form:\n"
                      "pytest tests/{suite}::TEST_NAME -q\n"
                      "Return only the command."),
                "b": ("[environment] Selection recorded.\n"
                      "Emit the exact command that runs only the test you "
                      "will address first, in the form:\n"
                      "pytest tests/{suite}::TEST_NAME -q\n"
                      "Return only the command."),
            },
            validator_id="v_test_command",
            max_new_tokens=48,
            validator_payload_by_sem={
                "a": {"selected": "{test_a}", "other": "{test_b}"},
                "b": {"selected": "{test_b}", "other": "{test_a}"},
            },
        ),
    ),
    # 7. storage format ------------------------------------------------------
    ScenarioSpec(
        scenario_id="arb_storage",
        bank="B-ARB3", family="ARB",
        contrast_axis="jsonl_vs_csv",
        semantic_a_id="jsonl", semantic_b_id="csv",
        framing_templates=(
            "{project} needs a flat-file export of the {table} table "
            "({domain}). Downstream tooling reads both supported layouts "
            "equally well, and file sizes are comparable.",
            "An export of {table} is being added to {project}. Both "
            "supported file layouts are read natively by every consumer "
            "and occupy similar space.",
        ),
        option_templates_a=(
            "Write the export as JSON Lines: one JSON object per record "
            "line.",
            "Use the JSON Lines layout — each record is one JSON object "
            "on its own line.",
        ),
        option_templates_b=(
            "Write the export as CSV: comma-separated rows under a "
            "header line.",
            "Use the CSV layout — records are comma-separated rows below "
            "one header.",
        ),
        ro_framing_templates=(
            "One question concerns the shape of a plain-text data dump "
            "when every reader handles both shapes.",
            "This entry asks about the on-disk layout for a tabular "
            "extract with no consumer constraint.",
        ),
        ro_option_templates_a=(
            "Emit each record as a self-describing bracketed structure "
            "on its own line.",
            "Store rows as independent structured objects, one per line, "
            "carrying their field names.",
        ),
        ro_option_templates_b=(
            "Emit records as delimited value rows beneath a single "
            "column-name line.",
            "Store rows as bare separated values with the field names "
            "given once up top.",
        ),
        incidentals=ARB3_INCIDENTALS,
        binding=BindingSpec(
            binding_kind="model_microtask",
            continuation_template_by_sem={
                "a": ("[environment] Selection recorded.\n"
                      "Emit the exact export command for the selected "
                      "layout, in the form:\n"
                      "convert {datafile} --format FORMAT\n"
                      "(FORMAT is jsonl or csv). Return only the command."),
                "b": ("[environment] Selection recorded.\n"
                      "Emit the exact export command for the selected "
                      "layout, in the form:\n"
                      "convert {datafile} --format FORMAT\n"
                      "(FORMAT is jsonl or csv). Return only the command."),
            },
            validator_id="v_format_command",
            max_new_tokens=48,
            validator_payload_by_sem={
                "a": {"format": "jsonl", "other": "csv"},
                "b": {"format": "csv", "other": "jsonl"},
            },
        ),
    ),
    # 8. traversal -----------------------------------------------------------
    ScenarioSpec(
        scenario_id="arb_traversal",
        bank="B-ARB3", family="ARB",
        contrast_axis="depth_first_vs_breadth_first",
        semantic_a_id="depth_first", semantic_b_id="breadth_first",
        framing_templates=(
            "{project}'s {module} must walk the content tree to build "
            "its index ({domain}). The tree fits in memory either way "
            "and both walk orders produce the same final index.",
            "To build its index, the {module} in {project} walks a "
            "content tree. Memory is ample for either order, and the "
            "finished index is identical under both.",
        ),
        option_templates_a=(
            "Walk the tree depth-first, finishing each subtree before "
            "moving on.",
            "Traverse depth-first: complete every subtree before its "
            "siblings.",
        ),
        option_templates_b=(
            "Walk the tree breadth-first, finishing each level before "
            "going deeper.",
            "Traverse breadth-first: complete every level before "
            "descending.",
        ),
        ro_framing_templates=(
            "One question concerns visiting order over a nested "
            "structure when the end result does not depend on it.",
            "This entry asks how to sweep a hierarchy when both sweeps "
            "land on the same answer.",
        ),
        ro_option_templates_a=(
            "Follow each branch to its end before backing out to the "
            "next branch.",
            "Go deep along one limb at a time, returning only when the "
            "limb is exhausted.",
        ),
        ro_option_templates_b=(
            "Sweep across each layer fully before dropping to the layer "
            "beneath.",
            "Go wide across one tier at a time, descending only when the "
            "tier is exhausted.",
        ),
        incidentals=ARB3_INCIDENTALS,
        binding=BindingSpec(
            binding_kind="model_microtask",
            continuation_template_by_sem={
                "a": ("[environment] Selection recorded.\n"
                      "Emit the exact indexing command for the selected "
                      "walk, in the form:\n"
                      "index {project} --strategy STRATEGY\n"
                      "(STRATEGY is depth-first or breadth-first). "
                      "Return only the command."),
                "b": ("[environment] Selection recorded.\n"
                      "Emit the exact indexing command for the selected "
                      "walk, in the form:\n"
                      "index {project} --strategy STRATEGY\n"
                      "(STRATEGY is depth-first or breadth-first). "
                      "Return only the command."),
            },
            validator_id="v_traversal_command",
            max_new_tokens=48,
            validator_payload_by_sem={
                "a": {"strategy": "depth-first", "other": "breadth-first"},
                "b": {"strategy": "breadth-first", "other": "depth-first"},
            },
        ),
    ),
    # 9. benchmark seed ------------------------------------------------------
    ScenarioSpec(
        scenario_id="arb_seed",
        bank="B-ARB3", family="ARB",
        contrast_axis="seed_zero_vs_seed_one",
        semantic_a_id="seed_zero", semantic_b_id="seed_one",
        framing_templates=(
            "A benchmark of {project}'s {module} is being scheduled "
            "({domain}). The harness needs one fixed random seed; runs "
            "are equally valid under any seed and take the same time.",
            "One benchmark run of {module} in {project} is queued. A "
            "single fixed seed must be supplied; every seed gives an "
            "equally valid, equally long run.",
        ),
        option_templates_a=(
            "Run the benchmark with seed 0.",
            "Use seed 0 for the benchmark run.",
        ),
        option_templates_b=(
            "Run the benchmark with seed 1.",
            "Use seed 1 for the benchmark run.",
        ),
        ro_framing_templates=(
            "One question concerns fixing an arbitrary constant for a "
            "repeatable measurement.",
            "This entry asks which starting constant to lock in for a "
            "reproducible timing run.",
        ),
        ro_option_templates_a=(
            "Anchor the run on the number zero.",
            "Lock the starting constant to zero.",
        ),
        ro_option_templates_b=(
            "Anchor the run on the number one.",
            "Lock the starting constant to one.",
        ),
        incidentals=ARB3_INCIDENTALS,
        binding=BindingSpec(
            binding_kind="model_microtask",
            continuation_template_by_sem={
                "a": ("[environment] Selection recorded.\n"
                      "Emit the exact benchmark command using the "
                      "selected seed, in the form:\n"
                      "bench run --suite {suite} --seed N --trials 20\n"
                      "Return only the command."),
                "b": ("[environment] Selection recorded.\n"
                      "Emit the exact benchmark command using the "
                      "selected seed, in the form:\n"
                      "bench run --suite {suite} --seed N --trials 20\n"
                      "Return only the command."),
            },
            validator_id="v_seed_command",
            max_new_tokens=48,
            validator_payload_by_sem={
                "a": {"seed": "0", "other": "1"},
                "b": {"seed": "1", "other": "0"},
            },
        ),
    ),
    # 10. meeting-note order -------------------------------------------------
    ScenarioSpec(
        scenario_id="arb_notes",
        bank="B-ARB3", family="ARB",
        contrast_axis="actions_first_vs_decisions_first",
        semantic_a_id="actions_first", semantic_b_id="decisions_first",
        framing_templates=(
            "The {project} weekly sync notes ({domain}) need their two "
            "standard blocks written up: action items and decisions. "
            "Readers consult both equally; the blocks are independent.",
            "Write-up time for the {project} weekly sync: the notes get "
            "an action-item block and a decision block. Both are read "
            "equally often and neither refers to the other.",
        ),
        option_templates_a=(
            "Put the action-item block first, decisions below it.",
            "Lead with the action items and place decisions after.",
        ),
        option_templates_b=(
            "Put the decision block first, action items below it.",
            "Lead with the decisions and place action items after.",
        ),
        ro_framing_templates=(
            "One question concerns arranging two equally consulted "
            "segments of a recurring summary.",
            "This entry asks about the top-to-bottom order of two "
            "sections in a routine write-up.",
        ),
        ro_option_templates_a=(
            "Open the summary with the list of tasks people took away.",
            "Start the write-up with who is doing what next.",
        ),
        ro_option_templates_b=(
            "Open the summary with the list of calls the group settled.",
            "Start the write-up with what the group concluded.",
        ),
        incidentals=ARB3_INCIDENTALS,
        binding=_env("the action-item block", "the decision block"),
    ),
    # 11. shard order --------------------------------------------------------
    ScenarioSpec(
        scenario_id="arb_shard",
        bank="B-ARB3", family="ARB",
        contrast_axis="shard_north_vs_shard_south",
        semantic_a_id="shard_north_first", semantic_b_id="shard_south_first",
        framing_templates=(
            "{project}'s nightly rebuild of {table} ({domain}) covers "
            "two equal shards named north and south. They rebuild "
            "independently, take the same time, and both finish tonight.",
            "Tonight {project} rebuilds the {table} data in two same-size "
            "shards, labeled north and south. The shards are independent "
            "and equally fast; both complete before morning.",
        ),
        option_templates_a=(
            "Rebuild the north shard first, then the south shard.",
            "Start the rebuild with north and follow with south.",
        ),
        option_templates_b=(
            "Rebuild the south shard first, then the north shard.",
            "Start the rebuild with south and follow with north.",
        ),
        ro_framing_templates=(
            "One question concerns sequencing two same-size partitions "
            "of an overnight job.",
            "This entry asks which of two identical halves of a nightly "
            "task to run first.",
        ),
        ro_option_templates_a=(
            "Take the partition carrying the upper compass label before "
            "its partner.",
            "Begin with the half named for the top of the map.",
        ),
        ro_option_templates_b=(
            "Take the partition carrying the lower compass label before "
            "its partner.",
            "Begin with the half named for the bottom of the map.",
        ),
        incidentals=ARB3_INCIDENTALS,
        binding=_env("the north shard rebuild", "the south shard rebuild"),
    ),
    # 12. lint priority ------------------------------------------------------
    ScenarioSpec(
        scenario_id="arb_lint",
        bank="B-ARB3", family="ARB",
        contrast_axis="import_order_vs_line_length",
        semantic_a_id="import_order_first", semantic_b_id="line_length_first",
        framing_templates=(
            "{project}'s linter ({domain}) reports two rule families "
            "with equal counts in {module}: import-order violations and "
            "line-length violations. Both families get cleaned today, "
            "one family at a time.",
            "Two equally sized lint families remain in {module} of "
            "{project}: import-order issues and line-length issues. The "
            "cleanup handles one family at a time and finishes today.",
        ),
        option_templates_a=(
            "Clean the import-order violations first, then line length.",
            "Handle the import-order family before the line-length "
            "family.",
        ),
        option_templates_b=(
            "Clean the line-length violations first, then import order.",
            "Handle the line-length family before the import-order "
            "family.",
        ),
        ro_framing_templates=(
            "One question concerns which of two equal piles of style "
            "findings to clear first.",
            "This entry asks about ordering two same-size batches of "
            "mechanical code cleanup.",
        ),
        ro_option_templates_a=(
            "Clear the findings about how file headers arrange their "
            "included modules.",
            "First fix where the top-of-file inclusion statements sit "
            "relative to each other.",
        ),
        ro_option_templates_b=(
            "Clear the findings about lines running past the width "
            "limit.",
            "First fix the statements that overrun the allowed column "
            "width.",
        ),
        incidentals=ARB3_INCIDENTALS,
        binding=_env("the import-order cleanup", "the line-length cleanup"),
    ),
)

assert len(ARB3_SCENARIOS) == 12
_MICRO = [s for s in ARB3_SCENARIOS
          if s.binding and s.binding.binding_kind == "model_microtask"]
assert len(_MICRO) == 6, [s.scenario_id for s in _MICRO]
