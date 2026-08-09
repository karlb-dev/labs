"""Lab 38 Phase 2 package: surface policy, semantic defaults, contextual
choice value, enacted choice, and report coupling.

Governing documents (precedence low->high): Phase 1 record (immutable),
Lab 38 handout, Phase 1 plan/addendum, ``preference/plans/preference_2_2.md``,
``preference/plans/preference_2_2_addendum.md`` (whose §B errata govern on
conflict). Claim ceiling (plan §8) applies to every artifact this package
writes, including log lines and commit messages.

Layout mirrors the campaign contract (plan §5):

- instrument identity: ``canonical`` (hashing), ``schema`` (data contracts)
- bank v3: ``scenarios`` (authored content), ``banks`` (deterministic
  expansion), ``formats`` (F-SYM / F-P1 / F-COMMIT), ``codebooks``
  (families + audits), ``lexical`` (balance audits)
- response contract: ``parser`` (strict), ``binding`` (branch resolution;
  E7 rules), ``targets``-equivalent logic inside ``codebooks``/``ports``
- execution: ``models`` (pins), ``modeling`` (pinned loading), ``chat``
  (per-model templates + shims), ``runner`` (resumable battery),
  ``scoring`` (single-row margins), ``capture`` (manifested activations)
- analysis: ``surface_analysis`` (B-SURF + Phase 1 reconstruction),
  ``behavioral_analysis`` (folded margins, sign-flip, criteria),
  ``power`` (simulation), ``mechanism`` (context-fitted directions,
  prechecks, interventions), ``coupling`` (RO readout + AR->RO)
- evidence: ``registry`` (append-only events), ``provenance``,
  ``artifacts``, ``reporting``, ``figures``, ``language_wall``
"""

__version__ = "0.1.0"

SCHEMA_VERSION = 2
BANK_VERSION = "pref2_v3"
EVENT_PREFIX = "pref2-"
STUDY_ID = "preference-phase2"

# Closed scientific-tier vocabulary (inherited; plan §7). Anything else is
# rejected.
SCIENTIFIC_TIERS = frozenset(
    {"instrument", "development", "frozen_behavioral", "conditional_causal"}
)

# Result taxonomy (plan §4) — closed vocabulary for adjudication artifacts.
RESULT_STATUSES = frozenset({
    "INSTRUMENT_FAILURE",
    "SURFACE_POLICY_ONLY",
    "SEMANTIC_MARGIN",
    "ENACTED_CHOICE",
    "CONTEXTUAL_VALUE",
    "MARGIN_HANDLE",
    "DIRECT_OUTPUT",
    "BEHAVIOR_SPECIFIC",
    "REPORT_SPECIFIC",
    "CHOICE_REPORT_COUPLED",
    "CLEAN_NULL",
})
