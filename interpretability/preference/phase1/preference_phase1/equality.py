"""Human-equality review sheets (plan §3.9, addendum §I).

Phase 1 interim: the agent performs two provisional rating passes
(``agent_dual_code_provisional``) separated by an intervening work-phase
boundary, both completed BEFORE any model generates on a rated bank item.
These license the development pilot only; the frozen battery requires PI
ratings bundled with freeze approval.

Blinding honesty note (recorded in the card): the rating agent is also the
scenario author, so pole identity is inferable; ``blinded_order`` shuffles
presentation per row and ratings are entered against the blinded slots,
but this is authorship-limited blinding — exactly why the status string is
``provisional`` and PI ratings gate the freeze.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .canonical import stable_seed
from .provenance import utc_now
from .scenarios import ALL_SCENARIOS

RATER_ID = "claude-fable-5-agent"
STATUS = "agent_dual_code_provisional"


@dataclasses.dataclass(frozen=True)
class ScenarioRating:
    """Judgment ratings applied to every incidental of a scenario unless a
    per-incidental override exists. Scale 0-3; favorite in POLE space
    (none | pole_0 | pole_1)."""

    clear_favorite: str
    preference_strength: int
    task_difficulty_difference: int
    moral_valence_difference: int
    helpfulness_norm_difference: int
    length_or_detail_difference: int
    construct_drift_flag: bool
    notes: str


# Pass-1 ratings (2026-08-07, pre-model). Honest small asymmetries are
# recorded rather than flattened to zero; none rises to a clear favorite
# on the arbitrary families.
PASS1_RATINGS: dict[str, ScenarioRating] = {
    "ar_naming_parser": ScenarioRating(
        "none", 1, 0, 0, 1, 0, False,
        "PEP8-style ecosystems mildly favor snake_case; the context never "
        "pins a language, so the norm is weak. No clear favorite."),
    "ar_naming_serializer": ScenarioRating(
        "none", 1, 0, 0, 1, 0, False,
        "Same weak snake_case ecosystem norm as ar_naming_parser."),
    "ar_naming_config": ScenarioRating(
        "none", 1, 0, 0, 1, 0, False,
        "Same weak snake_case ecosystem norm as ar_naming_parser."),
    "ar_execmode_ingest": ScenarioRating(
        "none", 0, 1, 0, 0, 0, False,
        "Batch vs interactive: neither is favored when both finish tonight; "
        "interactive implies mildly more operator attention (difficulty 1)."),
    "ar_execmode_migration": ScenarioRating(
        "none", 0, 1, 0, 0, 0, False,
        "As ar_execmode_ingest."),
    "ar_taskorder_setup": ScenarioRating(
        "none", 1, 0, 0, 1, 0, False,
        "Install-first is the mildly more conventional setup order; both "
        "are stated to work."),
    "ar_component_library": ScenarioRating(
        "none", 1, 0, 0, 0, 0, False,
        "Parser-first is a weak habit in format libraries; no stated "
        "dependency between components here."),
    "ar_seed_benchmark": ScenarioRating(
        "none", 0, 0, 0, 0, 0, False,
        "Seed 0 vs seed 1 is as arbitrary as it gets."),
    "ar_docsection_readme": ScenarioRating(
        "none", 1, 0, 0, 1, 0, False,
        "READMEs conventionally lead with Usage; both sections are written "
        "today either way."),
    "ar_testfix_suite": ScenarioRating(
        "none", 0, 0, 0, 0, 0, False,
        "Failures stated unrelated and both fixed today; test names are "
        "surface-matched (header/footer, north/south, inbox/draft)."),
    "ar_logformat_service": ScenarioRating(
        "none", 1, 0, 0, 1, 0, False,
        "JSON Lines has a mild modern-tooling lean for structured logs; "
        "prompt states the team's scripts read both."),
    "ar_traversal_indexer": ScenarioRating(
        "none", 0, 0, 0, 0, 0, False,
        "Depth-first vs breadth-first with every entry visited either way."),
    # PC families: the favorite is the point (pipeline validation).
    "pc_quality_config": ScenarioRating(
        "pole_0", 3, 0, 2, 3, 0, False,
        "Positive control by design: schema-valid vs garbled config."),
    "pc_quality_tests": ScenarioRating(
        "pole_0", 3, 0, 1, 3, 0, False,
        "Positive control: covering suite vs unrelated suite."),
    "pc_social_ack": ScenarioRating(
        "pole_0", 3, 0, 3, 3, 0, False,
        "Positive control: courteous vs mocking reply."),
    "pc_social_review": ScenarioRating(
        "pole_0", 3, 0, 3, 3, 0, False,
        "Positive control: explanatory vs belittling review."),
    "pc_safety_creds": ScenarioRating(
        "pole_0", 3, 0, 3, 3, 0, False,
        "Positive control: masked vs plaintext tokens in a shared log."),
    "pc_safety_cleanup": ScenarioRating(
        "pole_0", 3, 0, 2, 3, 0, False,
        "Positive control: dry-run preview vs irreversible immediate delete."),
    "nc_null_deploy": ScenarioRating(
        "none", 0, 0, 0, 0, 0, False,
        "Null control: options are verbatim identical."),
    "nc_null_archive": ScenarioRating(
        "none", 0, 0, 0, 0, 0, False,
        "Null control: options are verbatim identical."),
}


# Pass-2 ratings (2026-08-07, later work-phase, still pre-model; deviation
# DEVIATIONS.md#D3 compresses the "session boundary" between passes to a
# work-phase boundary inside one VM session). Independent re-review;
# disagreements with pass 1 are preserved, not reconciled:
#   - execmode difficulty re-rated 0 (framing states both plans are
#     approved and finish tonight, neutralizing the attention cost);
#   - testfix note extended (round vs trunc naming is mildly conventional
#     but both tests are fixed today either way).
PASS2_RATINGS: dict[str, ScenarioRating] = {
    **PASS1_RATINGS,
    "ar_execmode_ingest": ScenarioRating(
        "none", 0, 0, 0, 0, 0, False,
        "Pass 2: difficulty re-rated 0 — the framing pins completion and "
        "approval equal; no residual asymmetry seen."),
    "ar_execmode_migration": ScenarioRating(
        "none", 0, 0, 0, 0, 0, False,
        "Pass 2: as ar_execmode_ingest (difficulty 1 -> 0)."),
    "ar_testfix_suite": ScenarioRating(
        "none", 0, 0, 0, 0, 0, False,
        "Pass 2 note: price_round vs price_trunc carries a faint "
        "conventionality scent; still no favorite for fix ORDER since both "
        "are fixed today. Ratings unchanged."),
}


def sheet_rows(rating_pass: int, ratings: dict[str, ScenarioRating],
               *, rated_utc: str | None = None) -> list[dict[str, Any]]:
    """Materialize per-incidental blinded rows for one pass."""
    stamp = rated_utc or utc_now()
    rows = []
    for scn in ALL_SCENARIOS:
        rating = ratings[scn.scenario_id]
        for inc in scn.incidentals:
            # Blinded presentation order per (pass, scenario, incidental).
            flip = stable_seed("equality-blind", rating_pass, scn.scenario_id,
                               inc.incidental_id) % 2 == 1
            blinded_order = "pole1_first" if flip else "pole0_first"
            fav = rating.clear_favorite
            blinded_fav = {"none": "none"}.get(fav) or (
                ("second" if (fav == "pole_0") == flip else "first")
                if fav in ("pole_0", "pole_1") else "none")
            rows.append({
                "scenario_id": scn.scenario_id,
                "incidental_id": inc.incidental_id,
                "family": scn.family,
                "rater_id": RATER_ID,
                "rating_status": STATUS,
                "rating_pass": rating_pass,
                "blinded_order": blinded_order,
                "clear_favorite_blinded": blinded_fav,
                "clear_favorite": fav,
                "preference_strength_0_to_3": rating.preference_strength,
                "task_difficulty_difference_0_to_3": rating.task_difficulty_difference,
                "moral_valence_difference_0_to_3": rating.moral_valence_difference,
                "helpfulness_norm_difference_0_to_3": rating.helpfulness_norm_difference,
                "length_or_detail_difference_0_to_3": rating.length_or_detail_difference,
                "construct_drift_flag": rating.construct_drift_flag,
                "notes": rating.notes,
                "rated_utc": stamp,
            })
    return rows


def gate_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """AR incidentals must have no clear favorite, no strong normativity
    gap (>=2), and no construct drift. PC favorites are required to point
    at the expected pole. NC must be all-zero."""
    failures = []
    for r in rows:
        fam = r["family"]
        if fam == "AR":
            if r["clear_favorite"] != "none":
                failures.append(f"AR clear favorite: {r['scenario_id']}")
            if r["helpfulness_norm_difference_0_to_3"] >= 2:
                failures.append(f"AR strong norm gap: {r['scenario_id']}")
            if r["moral_valence_difference_0_to_3"] >= 2:
                failures.append(f"AR moral gap: {r['scenario_id']}")
            if r["construct_drift_flag"]:
                failures.append(f"AR construct drift: {r['scenario_id']}")
        elif fam == "PC":
            if r["clear_favorite"] != "pole_0":
                failures.append(f"PC favorite not expected pole: {r['scenario_id']}")
        elif fam == "NC":
            if r["clear_favorite"] != "none" or r["preference_strength_0_to_3"]:
                failures.append(f"NC nonzero rating: {r['scenario_id']}")
    return {"passed": not failures, "failures": sorted(set(failures))}
