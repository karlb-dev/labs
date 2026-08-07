# RETROSPECTIVE_ANALYSIS_RULES.md — allowed and forbidden analyses (P0, frozen)

Predeclared per `paper_analysis.md` §3.2 before any cross-study join.

## Allowed without a new registered event

- exact reconstruction of registered analyses;
- descriptive juxtaposition of registered estimates;
- claim-status and evidence-tier analysis;
- retrospective plots of already registered quantities;
- sensitivity analyses already frozen or contained in source outputs;
- new descriptive cross-study tables with no inferential promotion;
- meta-data analysis of reproducibility, missingness, and instrument
  gates.

## Requires a new registered retrospective-synthesis event before use in a paper claim

- a new bootstrap or model not present in any source analysis;
- a newly chosen threshold;
- a new subgroup split;
- a new cross-model statistical test;
- a new causal or moderator interpretation;
- any row-level join that changes the unit of inference.

Registration mechanism: an `analysis-*` event appended to
`interpretability/jspaces/phases/paper_analysis/analysis/reports/analysis_events.jsonl`
(this phase's own event log — campaign registries stay read-only), with
question, inputs (evidence IDs + hashes), method, and retrospective label,
committed **before** the analysis output is cited by any paper-facing
document.

## Forbidden

- inspecting sealed confirmatory outcomes that were never opened;
- re-running model outputs (any forward or backward pass);
- choosing paper endpoints by whichever post-hoc aggregate looks
  strongest;
- treating development and confirmatory rows as exchangeable;
- using capability-gated rows as zeros;
- fitting a convergence rate from three nested Qwen increments;
- claiming capacity causes causal organization from a handful of models;
- claiming training causes the OLMo pattern from a natural wedge;
- inferring a named Gemma mechanism from the Stage-1 mismatch alone;
- reopening Q-L4, restoring a retired primary, or writing into any
  campaign registry or registered output path;
- new bank/family authoring;
- handout regeneration before the P6 route decision;
- smuggling GPU work in as "analysis" — anything requiring a forward or
  backward pass goes on the Phase 5 horizon list (addendum §2.6).

## Missing-data doctrine

`missing`, `gated`, and `not_applicable` are data states with named
reasons (see `EVIDENCE_TIER_RULES.md` rule 3). The unsupported-number
register (`reports/UNSUPPORTED_NUMBER_REGISTER.md`) is published with the
papers, empty or not.
