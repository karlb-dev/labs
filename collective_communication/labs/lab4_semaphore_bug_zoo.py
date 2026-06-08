"""Lab 4: semaphore bug zoo.

This lab teaches synchronization failure modes around TPU remote DMA without
turning the normal course run into a hang-prone goblin cave.

There are two ideas in this file:

1. A *correct probe* that reuses the Lab 1 single-hop remote-copy kernel. This
   gives students one small, runnable Pallas program that exercises the entry
   barrier plus DMA send/receive semaphores.
2. A *bug zoo* catalog. Most broken variants are intentionally not executed by
   the benchmark harness because a real over-wait can hang the TPU process and
   force a restart. Instead, the catalog records the invariant, mutation,
   expected symptom, diagnostic, prevention rule, and recovery rule.

The benchmark harness owns timing, run directories, JSON/CSV output, plotting,
and profiler capture. This file owns the teaching content for the lab.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


# -----------------------------------------------------------------------------
# Public constants used by the lab handout and benchmark artifacts.
# -----------------------------------------------------------------------------

LAB_ID = "lab4"
LAB_TITLE = "Semaphore Bug Zoo"
CATALOG_VERSION = 2
CORRECT_PROBE_OP = "pallas_semaphore_correct"
BUG_ZOO_OP = "semaphore_bug_zoo"

CATEGORY_DMA = "dma"
CATEGORY_BARRIER = "barrier"
CATEGORY_REGULAR = "regular_semaphore"
CATEGORY_COLLECTIVE_ID = "collective_id"
CATEGORY_TOPOLOGY = "topology"
CATEGORY_BUFFER_OWNERSHIP = "buffer_ownership"
CATEGORY_PIPELINE = "pipeline"

SAFETY_CORRECTNESS_ONLY = "safe_correctness_failure"
SAFETY_HANG = "danger_hang"
SAFETY_CRASH = "danger_crash_or_nonzero_semaphore"
SAFETY_RACE = "danger_race_or_silent_corruption"
SAFETY_ISOLATED_ONLY = "danger_isolated_repro_only"

LAB4_INVARIANTS: tuple[str, ...] = (
    "Every remote DMA has exactly one intended sender and one intended receiver.",
    "Every DMA wait matches the number of bytes that can actually arrive.",
    "Every DMA descriptor is waited before its source or destination is reused.",
    "Every regular or barrier semaphore drains before kernel completion.",
    "Every cross-device phase has an entry synchronization story.",
    "Every collective_id names one compatible communication pattern.",
    "Every destination buffer slot has one writer at a time.",
    "Every benchmarked communication mutation has a correctness oracle.",
)


@dataclasses.dataclass(frozen=True)
class BugScenario:
    """One synchronization failure mode for the Lab 4 catalog.

    The core fields are ``id``, ``invariant``, ``mutation``,
    ``expected_symptom``, ``safe_to_run_by_default``, ``diagnostic``, and
    ``recovery``. Later fields add teaching metadata for richer JSON and Markdown
    artifacts.
    """

    id: str
    invariant: str
    mutation: str
    expected_symptom: str
    safe_to_run_by_default: bool
    diagnostic: str
    recovery: str
    title: str = ""
    category: str = "general"
    safety_level: str = SAFETY_ISOLATED_ONLY
    prevention: str = ""
    related_labs: tuple[str, ...] = ()
    worksheet_questions: tuple[str, ...] = ()
    profiler_clue: str = ""
    likely_student_mistake: str = ""
    isolated_repro_advice: str = (
        "Do not run this mutation in the default sweep. Use a disposable run, "
        "a fresh process, a small payload, and an external timeout."
    )

    @property
    def display_title(self) -> str:
        return self.title or self.id.replace("_", " ").title()

    @property
    def safety_badge(self) -> str:
        if self.safe_to_run_by_default:
            return "safe"
        if self.safety_level == SAFETY_HANG:
            return "hang risk"
        if self.safety_level == SAFETY_CRASH:
            return "crash risk"
        if self.safety_level == SAFETY_RACE:
            return "race risk"
        return "isolated only"

    @property
    def default_policy(self) -> str:
        if self.safe_to_run_by_default:
            return "May be exercised by the normal Lab 4 run."
        return "Catalog only in the normal Lab 4 run."

    def to_row(self) -> dict[str, Any]:
        """Return a JSON/CSV-friendly row for benchmark artifacts."""

        row = dataclasses.asdict(self)
        row["display_title"] = self.display_title
        row["safety_badge"] = self.safety_badge
        row["default_policy"] = self.default_policy
        row["dangerous"] = not self.safe_to_run_by_default
        row["operation"] = BUG_ZOO_OP
        row["lab"] = LAB_ID
        return row


# -----------------------------------------------------------------------------
# The catalog itself. Each scenario is a small teaching card.
# -----------------------------------------------------------------------------

BUG_SCENARIOS: tuple[BugScenario, ...] = (
    BugScenario(
        id="overwait_dma",
        title="Over-waited DMA semaphore",
        category=CATEGORY_DMA,
        safety_level=SAFETY_HANG,
        invariant="Every DMA semaphore wait must match bytes that are sent.",
        mutation="Receiver waits for more DMA bytes than any sender produces.",
        expected_symptom="Kernel hang or runtime wait failure.",
        safe_to_run_by_default=False,
        diagnostic="Compare sent byte count, waited byte count, and tile shape.",
        prevention=(
            "Compute byte counts from the same shape and dtype that define the "
            "remote-copy source and destination refs."
        ),
        recovery="Make the receive wait exactly match the async-copy byte count.",
        related_labs=("lab1", "lab2", "lab5"),
        likely_student_mistake=(
            "Changing tile shape or dtype in one place while leaving a manual "
            "semaphore wait amount unchanged."
        ),
        worksheet_questions=(
            "Which device is supposed to send these bytes?",
            "How many bytes does the receiver wait for?",
            "What shape and dtype determine that number?",
        ),
        profiler_clue=(
            "Trace may stop at or before a wait region, and the run may never "
            "produce a completed result row."
        ),
    ),
    BugScenario(
        id="undersend_dma",
        title="Under-sent DMA payload",
        category=CATEGORY_DMA,
        safety_level=SAFETY_HANG,
        invariant="The sender and receiver must agree on the destination size.",
        mutation="Sender copies a smaller slice than the receiver waits for.",
        expected_symptom="Indefinite wait for bytes that never arrive.",
        safe_to_run_by_default=False,
        diagnostic="Compare source ref slice, destination ref slice, and wait size.",
        prevention=(
            "Derive source and destination block specs from one helper, then "
            "assert actual_payload_bytes in the case metadata."
        ),
        recovery="Make the remote-copy source and destination slices the same size.",
        related_labs=("lab1", "lab5", "lab8"),
        likely_student_mistake=(
            "Sending only one chunk of a pipelined payload while waiting as if "
            "the whole tile had arrived."
        ),
        worksheet_questions=(
            "Is the sender copying a full tile or one chunk?",
            "Does the receiver wait for the chunk size or the full tile size?",
        ),
        profiler_clue="Look for a wait whose preceding copy started with a smaller ref slice.",
    ),
    BugScenario(
        id="oversignal_regular",
        title="Over-signaled regular or barrier semaphore",
        category=CATEGORY_REGULAR,
        safety_level=SAFETY_CRASH,
        invariant="Every regular or barrier semaphore drains before completion.",
        mutation="Signal a semaphore more times than peers wait.",
        expected_symptom="Nonzero semaphore state at kernel completion.",
        safe_to_run_by_default=False,
        diagnostic="Check signal count minus wait count for every participant.",
        prevention=(
            "Maintain a signal/wait ledger for every phase and every device. "
            "The balance must return to zero."
        ),
        recovery="Balance signal and wait counts or use a fresh collective ID.",
        related_labs=("lab1", "lab4", "lab8"),
        likely_student_mistake=(
            "Adding a barrier signal to the right neighbor but forgetting that "
            "the wait is on the local semaphore signaled by the left neighbor."
        ),
        worksheet_questions=(
            "Who increments this semaphore?",
            "Who decrements it?",
            "What is the final value on every device?",
        ),
        profiler_clue="The kernel may finish its useful work but fail at completion.",
    ),
    BugScenario(
        id="overwait_regular",
        title="Over-waited regular semaphore",
        category=CATEGORY_REGULAR,
        safety_level=SAFETY_HANG,
        invariant="A regular semaphore wait must be matched by a peer signal.",
        mutation="Wait for a regular semaphore signal that no peer sends.",
        expected_symptom="Kernel hang on semaphore_wait.",
        safe_to_run_by_default=False,
        diagnostic="Draw a table of which rank signals which rank, then compare waits.",
        prevention="Use one ledger row per participant, not one row per code branch.",
        recovery="Add the missing signal, reduce the wait count, or remove the phase.",
        related_labs=("lab4", "lab8", "lab10"),
        likely_student_mistake="All ranks wait, but only a subset of ranks can signal.",
        worksheet_questions=(
            "Which ranks call semaphore_signal?",
            "Which ranks call semaphore_wait?",
            "Are those sets symmetric?",
        ),
        profiler_clue="Trace may show a partial custom op and no following benchmark samples.",
    ),
    BugScenario(
        id="missing_entry_barrier",
        title="Missing phase-entry barrier",
        category=CATEGORY_BARRIER,
        safety_level=SAFETY_RACE,
        invariant="All peers enter a communication phase before remote writes.",
        mutation="Remove the phase-entry barrier before remote DMA starts.",
        expected_symptom="Race-dependent wrong data or intermittent failure.",
        safe_to_run_by_default=False,
        diagnostic="Look for remote writes before every receiver is known to be in the kernel.",
        prevention="Use a clear entry barrier for every cross-device kernel phase.",
        recovery="Add an entry barrier or prove the destination refs are already ready.",
        related_labs=("lab1", "lab2", "lab5"),
        likely_student_mistake="Assuming all devices enter the Pallas kernel at the same instant.",
        worksheet_questions=(
            "Which peers must have entered before this remote copy starts?",
            "Which collective_id protects that entry?",
        ),
        profiler_clue="Failures may disappear when profiling slows execution.",
    ),
    BugScenario(
        id="collective_id_reuse",
        title="Incompatible collective_id reuse",
        category=CATEGORY_COLLECTIVE_ID,
        safety_level=SAFETY_RACE,
        invariant="A collective ID names one compatible communication pattern.",
        mutation="Reuse one collective ID for incompatible phases, axes, or loops.",
        expected_symptom="Semaphore aliasing, stale state, wrong data, or hang.",
        safe_to_run_by_default=False,
        diagnostic="List every Pallas call, mesh axis, and phase using the same ID.",
        prevention="Allocate collective_id values from a written phase map.",
        recovery="Assign distinct IDs for distinct concurrent or incompatible phases.",
        related_labs=("lab2", "lab5", "lab9"),
        likely_student_mistake="Using collective_id=0 everywhere because the first kernel worked.",
        worksheet_questions=(
            "Which communication pattern does this collective_id name?",
            "Can any other phase overlap or alias with it?",
        ),
        profiler_clue="The same kernel works alone but fails when composed with another phase.",
    ),
    BugScenario(
        id="mismatched_collective_order",
        title="Mismatched phase order across ranks",
        category=CATEGORY_COLLECTIVE_ID,
        safety_level=SAFETY_HANG,
        invariant="All peers execute communication phases in one compatible order.",
        mutation="Some ranks enter phase A then B while others enter phase B then A.",
        expected_symptom="Deadlock, stale semaphore state, or swapped data.",
        safe_to_run_by_default=False,
        diagnostic="Write a per-rank timeline and check that phase numbers line up.",
        prevention="Keep phase order data-independent, or split patterns into separate calls.",
        recovery="Make every rank execute the same phase order before adding conditionals.",
        related_labs=("lab5", "lab7", "lab8"),
        likely_student_mistake="Branching communication order on local rank or local data.",
        worksheet_questions=(
            "What is rank 0 waiting for in phase A?",
            "What is rank 1 sending in phase A?",
            "Do the phase names mean the same thing on every rank?",
        ),
        profiler_clue="The trace may show ranks entering different named scopes before the stall.",
    ),
    BugScenario(
        id="two_writers_one_slot",
        title="Two remote writers for one destination slot",
        category=CATEGORY_BUFFER_OWNERSHIP,
        safety_level=SAFETY_RACE,
        invariant="Each destination buffer slot has one writer at a time.",
        mutation="Two devices remotely copy into the same destination slot.",
        expected_symptom="Nondeterministic ownership or wrong final value.",
        safe_to_run_by_default=False,
        diagnostic="Trace destination slot ownership for every sender and every hop.",
        prevention="Assign destination slots by source rank, hop, or chunk index.",
        recovery="Partition slots, add buffering, or serialize the writers.",
        related_labs=("lab5", "lab6", "lab8"),
        likely_student_mistake="Using output[0] as a convenient scratch slot for every incoming shard.",
        worksheet_questions=(
            "Who writes this slot during the phase?",
            "Can another device write it before the first value is consumed?",
        ),
        profiler_clue="Timing may look fine while correctness changes across runs.",
    ),
    BugScenario(
        id="wrong_neighbor_map",
        title="Wrong neighbor map",
        category=CATEGORY_TOPOLOGY,
        safety_level=SAFETY_CORRECTNESS_ONLY,
        invariant="Sender, receiver, and correctness model share one topology map.",
        mutation="Send to right neighbors while validating as if sends went left.",
        expected_symptom="Fast correctness failure with otherwise clean completion.",
        safe_to_run_by_default=True,
        diagnostic="Compare the permutation table against expected output ranks.",
        prevention="Derive neighbors and expected ranks from one shared helper.",
        recovery="Derive neighbors from one shared mesh-index helper.",
        related_labs=("lab1", "lab2", "lab5", "lab9"),
        likely_student_mistake="Fixing the kernel direction but forgetting to update the expected ranks.",
        worksheet_questions=(
            "For N=4, what should device 0 receive in a right-moving ring?",
            "What should it receive in a left-moving ring?",
        ),
        profiler_clue="The operation completes normally, but rank markers are rotated the wrong way.",
        isolated_repro_advice="Safe to use as an expected-fail correctness example.",
    ),
    BugScenario(
        id="missing_dma_wait_before_use",
        title="Missing receive wait before consuming data",
        category=CATEGORY_DMA,
        safety_level=SAFETY_RACE,
        invariant="Async copies must complete before their outputs are consumed.",
        mutation="Read or reduce a destination buffer before waiting on the DMA descriptor.",
        expected_symptom="Stale data, partial data, or race-dependent output.",
        safe_to_run_by_default=False,
        diagnostic="Search for reads or reductions between async_remote_copy start and wait_recv.",
        prevention="Put every consume operation after the receive-side wait or use a proven buffer protocol.",
        recovery="Wait before consuming the destination or use double buffering.",
        related_labs=("lab3", "lab5", "lab6", "lab8"),
        likely_student_mistake="Starting a copy, doing local compute, and accidentally reading the destination early.",
        worksheet_questions=(
            "Where is the first read of the destination buffer?",
            "What proves that bytes have arrived before that read?",
        ),
        profiler_clue="Race may appear only when profiling is disabled or payload size changes.",
    ),
    BugScenario(
        id="missing_wait_send_before_source_reuse",
        title="Missing send wait before reusing source slot",
        category=CATEGORY_DMA,
        safety_level=SAFETY_RACE,
        invariant="The source slot cannot be overwritten while the DMA send may still read it.",
        mutation="Start a remote copy, then overwrite or reuse the source buffer before wait_send.",
        expected_symptom="Receiver observes partly old or partly new data.",
        safe_to_run_by_default=False,
        diagnostic="Find writes to the source slot between remote_copy.start() and wait_send().",
        prevention="Treat wait_send as the proof that the source slot is reusable.",
        recovery="Call wait_send before source reuse or use a different source buffer slot.",
        related_labs=("lab5", "lab8"),
        likely_student_mistake="Using one scratch slot for every chunk in a pipelined loop.",
        worksheet_questions=(
            "When does the source slot become safe to overwrite?",
            "Which wait proves that?",
        ),
        profiler_clue="Destination corruption may track source-slot reuse, not receiver timing.",
    ),
    BugScenario(
        id="buffer_slot_run_ahead",
        title="Producer runs ahead and reuses a buffer slot",
        category=CATEGORY_PIPELINE,
        safety_level=SAFETY_RACE,
        invariant="A slot cannot be reused while a peer may still read it.",
        mutation="Pipeline producer reuses a slot without downstream completion.",
        expected_symptom="Correctness failures only at some chunk counts or sizes.",
        safe_to_run_by_default=False,
        diagnostic="Track producer and consumer epochs for each buffer slot.",
        prevention="Use per-slot capacity semaphores or enough buffering for allowed run-ahead.",
        recovery="Add per-slot semaphores or increase the number of buffers.",
        related_labs=("lab8", "lab9", "lab10"),
        likely_student_mistake="Assuming double buffering is enough even when devices can get more than one step apart.",
        worksheet_questions=(
            "What is the epoch number of this buffer slot?",
            "Can the producer lap the consumer?",
        ),
        profiler_clue="Only larger payloads or higher chunk counts trigger the race.",
    ),
    BugScenario(
        id="single_recv_semaphore_multiple_inflight",
        title="One receive semaphore reused by multiple in-flight DMAs",
        category=CATEGORY_DMA,
        safety_level=SAFETY_RACE,
        invariant="Independent in-flight DMAs need unambiguous completion tracking.",
        mutation="Reuse one receive semaphore for multiple simultaneous incoming copies.",
        expected_symptom="A wait may correspond to the wrong copy or merge progress from multiple copies.",
        safe_to_run_by_default=False,
        diagnostic="Count all in-flight DMAs that can increment the same receive semaphore.",
        prevention="Use separate receive semaphores or serialize copies that share one semaphore.",
        recovery="Allocate one receive semaphore per independent in-flight copy or phase.",
        related_labs=("lab5", "lab8"),
        likely_student_mistake="Adding pipelining while keeping the single-semaphore Lab 1 structure.",
        worksheet_questions=(
            "How many DMAs can be in flight at once?",
            "Can their completion signals be distinguished?",
        ),
        profiler_clue="The trace shows multiple remote copies active before a single receive wait.",
    ),
    BugScenario(
        id="device_id_type_or_axis_mismatch",
        title="Device-id type or mesh-axis mismatch",
        category=CATEGORY_TOPOLOGY,
        safety_level=SAFETY_ISOLATED_ONLY,
        invariant="The device_id passed to remote DMA must match the chosen ID type.",
        mutation="Pass a mesh tuple to code expecting a logical ID, or use the wrong mesh axis.",
        expected_symptom="Compile error, wrong target, missing receiver, or hang.",
        safe_to_run_by_default=False,
        diagnostic="Print the mesh shape, axis name, source-target pairs, and device_id_type.",
        prevention="Keep neighbor computation in one helper and document whether IDs are mesh coordinates or logical IDs.",
        recovery="Use the correct ID type and recompute expected ranks from the same map.",
        related_labs=("lab1", "lab5", "lab9", "lab10"),
        likely_student_mistake="Moving from a 1D ring to a 2D mesh while leaving axis_index logic wired to the old axis.",
        worksheet_questions=(
            "Is dst a scalar logical ID or a tuple of mesh coordinates?",
            "Which axis is being permuted?",
        ),
        profiler_clue="The failure may look like a missing sender even though a send was issued.",
    ),
)


# -----------------------------------------------------------------------------
# Catalog helpers consumed by the benchmark harness and the Markdown artifact.
# -----------------------------------------------------------------------------


def validate_catalog(scenarios: Sequence[BugScenario] = BUG_SCENARIOS) -> None:
    """Fail fast if the teaching catalog has internal inconsistencies."""

    ids = [scenario.id for scenario in scenarios]
    duplicates = sorted({scenario_id for scenario_id in ids if ids.count(scenario_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate Lab 4 scenario ids: {duplicates}")
    if not any(scenario.safe_to_run_by_default for scenario in scenarios):
        raise ValueError("at least one Lab 4 scenario should be safe to run by default")
    for scenario in scenarios:
        if not scenario.invariant.strip():
            raise ValueError(f"scenario {scenario.id!r} is missing an invariant")
        if not scenario.recovery.strip():
            raise ValueError(f"scenario {scenario.id!r} is missing a recovery rule")


def scenario_by_id(scenario_id: str) -> BugScenario:
    """Return one scenario by id, with a helpful error for typos."""

    for scenario in BUG_SCENARIOS:
        if scenario.id == scenario_id:
            return scenario
    known = ", ".join(sorted(scenario.id for scenario in BUG_SCENARIOS))
    raise KeyError(f"unknown Lab 4 scenario {scenario_id!r}; known ids: {known}")


def filtered_scenarios(
    *,
    include_dangerous: bool = True,
    category: str | None = None,
    safety_level: str | None = None,
) -> tuple[BugScenario, ...]:
    """Return scenarios matching optional teaching filters."""

    scenarios: Iterable[BugScenario] = BUG_SCENARIOS
    if not include_dangerous:
        scenarios = (s for s in scenarios if s.safe_to_run_by_default)
    if category is not None:
        scenarios = (s for s in scenarios if s.category == category)
    if safety_level is not None:
        scenarios = (s for s in scenarios if s.safety_level == safety_level)
    return tuple(scenarios)


def scenario_rows(
    *,
    include_dangerous: bool = True,
    category: str | None = None,
    safety_level: str | None = None,
) -> list[dict[str, Any]]:
    """Return JSON/CSV rows for the bug zoo artifact.

    Callable with no arguments to render the full catalog.
    """

    return [
        scenario.to_row()
        for scenario in filtered_scenarios(
            include_dangerous=include_dangerous,
            category=category,
            safety_level=safety_level,
        )
    ]


def safe_scenario_rows() -> list[dict[str, Any]]:
    """Return rows that are safe to exercise in the default lab run."""

    return scenario_rows(include_dangerous=False)


def dangerous_scenario_rows() -> list[dict[str, Any]]:
    """Return rows that should remain catalog-only by default."""

    return [scenario.to_row() for scenario in BUG_SCENARIOS if not scenario.safe_to_run_by_default]


def safe_scenarios() -> tuple[BugScenario, ...]:
    """Return scenario objects that may be exercised by the default Lab 4 run."""

    return filtered_scenarios(include_dangerous=False)


def dangerous_scenarios() -> tuple[BugScenario, ...]:
    """Return scenario objects that should remain catalog-only by default."""

    return tuple(s for s in BUG_SCENARIOS if not s.safe_to_run_by_default)


def catalog_summary(scenarios: Sequence[BugScenario] = BUG_SCENARIOS) -> dict[str, Any]:
    """Small summary dictionary for run artifacts."""

    by_category = Counter(s.category for s in scenarios)
    by_safety = Counter(s.safety_level for s in scenarios)
    safe = sum(1 for s in scenarios if s.safe_to_run_by_default)
    dangerous = len(scenarios) - safe
    return {
        "lab": LAB_ID,
        "title": LAB_TITLE,
        "operation": BUG_ZOO_OP,
        "catalog_version": CATALOG_VERSION,
        "total_scenarios": len(scenarios),
        "safe_to_run_by_default": safe,
        "catalog_only_by_default": dangerous,
        "by_category": dict(sorted(by_category.items())),
        "by_safety_level": dict(sorted(by_safety.items())),
        "invariants": list(LAB4_INVARIANTS),
    }


def scenario_counts(scenarios: Sequence[BugScenario] = BUG_SCENARIOS) -> dict[str, int]:
    """Compact count summary for tests and small dashboards."""

    summary = catalog_summary(scenarios)
    return {
        "total": int(summary["total_scenarios"]),
        "safe_to_run_by_default": int(summary["safe_to_run_by_default"]),
        "catalog_only_by_default": int(summary["catalog_only_by_default"]),
    }


def semaphore_ledger_template() -> list[dict[str, str]]:
    """Return the worksheet students should fill before writing a custom phase."""

    return [
        {
            "field": "phase_name",
            "question": "What communication phase is this?",
            "example": "lab5_all_gather_hop_2",
        },
        {
            "field": "collective_id",
            "question": "Which barrier semaphore namespace does this phase use?",
            "example": "base_collective_id + hop",
        },
        {
            "field": "participants",
            "question": "Which devices participate?",
            "example": "all devices on mesh axis x",
        },
        {
            "field": "source_ref",
            "question": "Which local ref or slot is read by the send?",
            "example": "send_buffer[working_slot, ...]",
        },
        {
            "field": "destination_ref",
            "question": "Which remote ref or slot is written?",
            "example": "recv_buffer[receiving_slot, ...] on right neighbor",
        },
        {
            "field": "bytes",
            "question": "How many bytes can be sent and received?",
            "example": "rows * cols * dtype.itemsize",
        },
        {
            "field": "send_wait",
            "question": "What proves the source can be reused?",
            "example": "remote_copy.wait_send()",
        },
        {
            "field": "recv_wait",
            "question": "What proves the destination can be consumed?",
            "example": "remote_copy.wait_recv()",
        },
        {
            "field": "buffer_owner_after_phase",
            "question": "Who owns each buffer slot after the phase?",
            "example": "device i owns shard from device i - hop",
        },
    ]


def render_json(scenarios: Sequence[BugScenario] = BUG_SCENARIOS, *, indent: int = 2) -> str:
    """Render the bug zoo as JSON for scripts, tests, and lab artifacts."""

    validate_catalog(scenarios)
    payload = {
        "lab": LAB_ID,
        "title": LAB_TITLE,
        "operation": BUG_ZOO_OP,
        "catalog_version": CATALOG_VERSION,
        "summary": catalog_summary(scenarios),
        "semaphore_ledger_template": semaphore_ledger_template(),
        "scenarios": [scenario.to_row() for scenario in scenarios],
    }
    return json.dumps(payload, indent=indent, sort_keys=True) + "\n"


def write_artifacts(
    artifact_dir: str | Path,
    *,
    prefix: str = LAB_ID,
    scenarios: Sequence[BugScenario] = BUG_SCENARIOS,
) -> dict[str, str]:
    """Write JSON and Markdown bug-zoo artifacts.

    The benchmark runner may already own artifact writing. This helper is here
    for smoke tests, notebooks, and future runners that want the lab module to
    produce its own human and machine-readable files.
    """

    validate_catalog(scenarios)
    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    safe_prefix = str(prefix).strip() or LAB_ID

    json_path = path / f"{safe_prefix}_semaphore_bug_zoo.json"
    md_path = path / f"{safe_prefix}_semaphore_bug_zoo.md"

    json_path.write_text(render_json(scenarios), encoding="utf-8")
    md_path.write_text(render_markdown(scenarios), encoding="utf-8")

    return {"json": str(json_path), "markdown": str(md_path)}


# -----------------------------------------------------------------------------
# Correct probe compatibility helpers.
# -----------------------------------------------------------------------------


def _load_lab1_module() -> Any:
    """Import Lab 1 from either the package layout or a local lab directory."""

    try:
        from labs import lab1_single_hop  # type: ignore
        return lab1_single_hop
    except ImportError:
        # This fallback is convenient when a student copies lab files into one
        # directory for experimentation. It is intentionally lazy so importing
        # Lab 4 only needs the standard library.
        import lab1_single_hop  # type: ignore
        return lab1_single_hop


def build_correct_probe_case(**kwargs: Any) -> Any:
    """Build the runnable Lab 4 correctness probe by delegating to Lab 1.

    The probe is intentionally the Lab 1 single-hop remote copy. Lab 4 is about
    the synchronization rules around that copy, so reusing the same tiny kernel
    keeps the experiment legible.
    """

    lab1_single_hop = _load_lab1_module()
    return lab1_single_hop.build_case(**kwargs)


def build_case(**kwargs: Any) -> Any:
    """Backward-friendly alias for the correct semaphore probe case."""

    return build_correct_probe_case(**kwargs)


def check_correct_probe_result(jax: Any, jnp: Any, y: Any, expected_ranks: Any) -> bool:
    """Validate the correct probe by delegating to Lab 1's checker."""

    lab1_single_hop = _load_lab1_module()
    return bool(lab1_single_hop.check_result(jax, jnp, y, expected_ranks))


def check_result(jax: Any, jnp: Any, y: Any, expected_ranks: Any) -> bool:
    """Backward-friendly alias for the correct semaphore probe checker."""

    return check_correct_probe_result(jax, jnp, y, expected_ranks)


# -----------------------------------------------------------------------------
# Runnable bug demos.
#
# The catalog above describes failure modes. These helpers let students actually
# *run* one and watch the documented symptom appear, instead of only reading
# about it.
#
# Only correctness-class bugs (those that complete cleanly and fail validation)
# are safe to run in-process and are exposed here. Hang/crash/race-class bugs are
# NOT shipped as runnable kernels: a real over-wait can deadlock the TPU process
# and force a runtime restart, so they must run in an isolated, disposable
# environment behind an external timeout. The benchmark harness provides that
# guarded subprocess path; see ``run_guarded_subprocess`` in ``collective_bench``
# and the "Run a real bug" section of ``lab4_semaphore_bug_zoo.md``.
# -----------------------------------------------------------------------------

# Bug ids that are safe to execute in-process: they complete normally and fail
# only the correctness check, reproducing the documented symptom without any risk
# of hanging or corrupting the device.
SAFE_RUNNABLE_BUGS: tuple[str, ...] = ("wrong_neighbor_map",)


def _flip_direction(direction: str) -> str:
    """Return the opposite ring direction using Lab 1's normalizer."""

    lab1 = _load_lab1_module()
    return "left" if lab1.normalize_direction(direction) == "right" else "right"


def build_wrong_neighbor_map_case(
    jax: Any,
    jnp: Any,
    *,
    intended_direction: str = "right",
    **kwargs: Any,
) -> tuple[Any, Any, str]:
    """Build a runnable, SAFE reproduction of the ``wrong_neighbor_map`` bug.

    The kernel is the correct Lab 1 single-hop copy, but built with the *opposite*
    neighbor direction from the one the caller intends. The kernel therefore runs
    to completion with no hang and no crash, yet the rank markers land on the
    wrong devices, so validation against the *intended* ownership map fails fast.
    That is exactly the catalog's documented symptom: "fast correctness failure
    with otherwise clean completion."

    Returns ``(case, intended_expected_ranks, buggy_direction)`` where ``case`` is
    a Lab 1 ``NeighborCopyCase`` whose ``expected_ranks`` reflect the *buggy*
    direction; ``intended_expected_ranks`` is what a correct kernel would have
    produced and is what the demo validates against to surface the mismatch.
    """

    lab1 = _load_lab1_module()
    intended = lab1.normalize_direction(intended_direction)
    buggy_direction = _flip_direction(intended)
    case = lab1.build_case(jax=jax, jnp=jnp, direction=buggy_direction, **kwargs)
    num_devices = len(kwargs["devices"])
    intended_expected = jnp.array(
        lab1.expected_neighbor_ranks(num_devices, intended),
        dtype=jnp.float32,
    )
    return case, intended_expected, buggy_direction


# -----------------------------------------------------------------------------
# Markdown rendering used by the run artifact.
# -----------------------------------------------------------------------------


def _md_cell(text: Any) -> str:
    """Escape a small value for a Markdown table cell."""

    return str(text).replace("|", "\\|").replace("\n", "<br>")


def _bullet_lines(items: Sequence[str]) -> list[str]:
    return [f"- {item}" for item in items]


def render_markdown(scenarios: Sequence[BugScenario] = BUG_SCENARIOS) -> str:
    """Render the bug zoo as a teaching artifact.

    Callable with no arguments to render the full catalog. The output includes a
    safety summary, scenario cards, and a debugging ledger students can reuse in
    later labs.
    """

    validate_catalog(scenarios)
    summary = catalog_summary(scenarios)
    safe_count = summary["safe_to_run_by_default"]
    dangerous_count = summary["catalog_only_by_default"]

    lines: list[str] = [
        "# Lab 4: Semaphore Bug Zoo",
        "",
        "This artifact catalogs synchronization failure modes for TPU Pallas ",
        "remote-DMA code. The normal lab run executes the correct semaphore ",
        "probe and writes this catalog. It does not execute hang-prone broken ",
        "kernels by default.",
        "",
        "## Summary",
        "",
        f"- Total scenarios: {summary['total_scenarios']}",
        f"- Safe to run by default: {safe_count}",
        f"- Catalog-only by default: {dangerous_count}",
        f"- Runnable correctness probe: `{CORRECT_PROBE_OP}`",
        f"- Catalog operation: `{BUG_ZOO_OP}`",
        "",
        "## Core Invariants",
        "",
        *_bullet_lines(LAB4_INVARIANTS),
        "",
        "## Scenario Table",
        "",
        "| id | category | safety | invariant | expected symptom |",
        "| --- | --- | --- | --- | --- |",
    ]

    for scenario in scenarios:
        lines.append(
            "| "
            f"`{scenario.id}` | "
            f"{_md_cell(scenario.category)} | "
            f"{_md_cell(scenario.safety_badge)} | "
            f"{_md_cell(scenario.invariant)} | "
            f"{_md_cell(scenario.expected_symptom)} |"
        )

    lines.extend(["", "## Scenario Cards", ""])
    for scenario in scenarios:
        lines.extend(
            [
                f"### `{scenario.id}`: {scenario.display_title}",
                "",
                f"- Category: `{scenario.category}`",
                f"- Safety: **{scenario.safety_badge}**",
                f"- Default policy: {scenario.default_policy}",
                f"- Invariant: {scenario.invariant}",
                f"- Mutation: {scenario.mutation}",
                f"- Expected symptom: {scenario.expected_symptom}",
                f"- Likely student mistake: {scenario.likely_student_mistake or 'Not specified.'}",
                f"- Diagnostic: {scenario.diagnostic}",
                f"- Prevention: {scenario.prevention or 'Not specified.'}",
                f"- Recovery: {scenario.recovery}",
                f"- Profiler clue: {scenario.profiler_clue or 'No reliable profiler clue.'}",
                f"- Isolated repro advice: {scenario.isolated_repro_advice}",
            ]
        )
        if scenario.related_labs:
            lines.append(f"- Related labs: {', '.join(scenario.related_labs)}")
        if scenario.worksheet_questions:
            lines.extend(["", "Worksheet questions:"])
            lines.extend(_bullet_lines(scenario.worksheet_questions))
        lines.append("")

    lines.extend(
        [
            "## Semaphore Ledger Template",
            "",
            "Students should fill this ledger before writing any multi-phase custom ",
            "collective. It turns vague synchronization into a checkable contract.",
            "",
            "| field | question | example |",
            "| --- | --- | --- |",
        ]
    )
    for row in semaphore_ledger_template():
        lines.append(
            "| "
            f"`{row['field']}` | "
            f"{_md_cell(row['question'])} | "
            f"{_md_cell(row['example'])} |"
        )

    lines.extend(
        [
            "",
            "## Safety Rule For This Lab",
            "",
            "Do not add dangerous broken kernels to the default benchmark sweep. A ",
            "real over-wait can hang the process. A real over-signal can crash at ",
            "kernel completion. Race-condition repros may corrupt data silently. ",
            "Dangerous variants belong in isolated subprocesses with explicit ",
            "timeouts, tiny payloads, and a disposable run environment.",
            "",
            "## Pass Condition",
            "",
            "```text",
            "the correct semaphore probe matches the Lab 1 ppermute ownership model",
            "the bug zoo artifact lists invariant, mutation, symptom, diagnostic, prevention, and recovery",
            "no hang-prone broken kernel runs by default",
            "students can fill the semaphore ledger for the next ring all-gather lab",
            "```",
            "",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "BUG_SCENARIOS",
    "BUG_ZOO_OP",
    "CORRECT_PROBE_OP",
    "LAB4_INVARIANTS",
    "LAB_ID",
    "LAB_TITLE",
    "BugScenario",
    "SAFE_RUNNABLE_BUGS",
    "build_case",
    "build_correct_probe_case",
    "build_wrong_neighbor_map_case",
    "catalog_summary",
    "check_correct_probe_result",
    "check_result",
    "dangerous_scenario_rows",
    "dangerous_scenarios",
    "filtered_scenarios",
    "render_json",
    "render_markdown",
    "safe_scenario_rows",
    "safe_scenarios",
    "scenario_by_id",
    "scenario_counts",
    "scenario_rows",
    "semaphore_ledger_template",
    "validate_catalog",
    "write_artifacts",
]


# Validate eagerly when imported by tests or the benchmark harness. This catches
# duplicate IDs before a run directory gets filled with confusing artifacts.
validate_catalog()


if __name__ == "__main__":  # pragma: no cover - convenience for humans.
    print(render_markdown())
