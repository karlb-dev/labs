"""Lab 38 Phase 1 package: stated vs revealed preference instrument.

Governing documents (precedence low->high): the Lab 38 draft handout,
``preference/plans/preference_1_1.md``, ``preference/plans/preference_1_1_addendum.md``.
Claim ceiling (plan §2.3) applies to every artifact this package writes,
including log lines and commit messages.

Layout mirrors the campaign contract:

- instrument identity: ``canonical`` (hashing), ``schema`` (data contracts)
- bank: ``scenarios`` (authored content), ``bank`` (deterministic expansion),
  ``lexical`` (balance audit), ``equality`` (human-equality review sheets)
- response contract: ``targets`` (codebook + tokenizer audit), ``parser``
  (strict/permissive), ``binding`` (branch resolution + follow-through)
- execution: ``modeling`` (pinned models via interp_bench-as-library),
  ``chat`` (template + boundary audit), ``runner`` (resumable battery)
- evidence: ``registry`` (append-only events), ``provenance`` (git/env),
  ``artifacts`` (atomic + durable writes), ``analysis`` (effects, gates),
  ``figures`` (registered plots), ``reporting`` (cards + TeX handout)
"""

__version__ = "0.1.0"

SCHEMA_VERSION = 1
BANK_VERSION = "lab38_v2_phase1"
EVENT_PREFIX = "pref1-"
STUDY_ID = "preference-phase1"

# Closed scientific-tier vocabulary (plan §1.5). Anything else is rejected.
SCIENTIFIC_TIERS = frozenset(
    {"instrument", "development", "frozen_behavioral", "conditional_causal"}
)
