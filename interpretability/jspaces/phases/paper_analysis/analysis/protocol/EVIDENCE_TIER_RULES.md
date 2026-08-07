# EVIDENCE_TIER_RULES.md — canonical tier vocabulary (P0, frozen)

Every evidence row, claim row, table cell, and figure panel in this
analysis carries **exactly one primary tier** from the canonical
vocabulary, plus optional status flags. Source tiers are preserved from
the originating registries; the analysis phase may **narrow** a tier
description but may never promote one.

## Canonical primary tiers

```text
confirmatory                 preregistered primary, gates passed, frozen partition
held_out_replication         frozen held-out families/partitions, prespecified
independent_lens_replication replication under an independently fitted lens
prespecified_development     prespecified but development-tier (side studies, Study 1/2)
methods                      instrument/measurement result, no scientific primary
exploratory                  Part-1-era or unpreregistered observation
retrospective_synthesis      produced by THIS analysis phase from frozen inputs
withdrawn                    formally withdrawn by registry event
superseded                   replaced by a later registered event
blocked                      blocked by external review or protocol gate
capability_gated             cohort empty/insufficient under prospective capability gate
not_applicable               removed from scope by a registered branch decision
```

## Status flags (non-exclusive)

```text
underpowered                 design evidence exists; power below frozen threshold
missing_not_zero             cell absent for a stated reason; never imputed as null
protocol_gated_missing       absent because a stop rule or gate fired
archive_unavailable          raw field absent from released artifacts
known_deficit                inside an accepted, named deficit (e.g. A120–A250 state.json)
```

## Binding rules

1. **No promotion.** A development-tier source result never becomes
   confirmatory by aggregation, juxtaposition, or narrative placement.
   Import events adopt the *native* tier of their source, never the tier
   of the importing registry.
2. **No pooling across estimands.** Rows with different estimator
   versions, protection geometries, or units of inference are never
   averaged; the canonical schemas make estimator and tier keys mandatory
   so this cannot happen silently.
3. **Missing, gated, and not-applicable are data states** (plan §1.5).
   The distinct states `observed zero`, `interval including zero`,
   `statistical equivalence`, `capability_gated missing`,
   `protocol-gated missing`, `not run by stop rule`, `not_applicable`,
   `underpowered`, `blocked`, and `archive_unavailable` are never
   collapsed into a single NA and never filled with zeros. Canonical
   example (enshrined): the OLMo SFT/DPO wedge capability table —
   0.617% / 0.309% direct-capability rates and **zero** Bank-S-capable
   cohort facts mean the stage cells are `capability_gated`, not null and
   not zero.
4. **Tier transitions must be visible.** Any figure or table mixing tiers
   marks each element's tier; development evidence is watermarked in
   figures where it appears beside confirmatory evidence.
5. **Six seed sentences enter the ledger at their frozen tiers**
   (addendum §2.3): (1) channels exist / content-not-dose —
   confirmatory + replicated; (2) training-dependent occupancy —
   prespecified_development, stage localization open missing-not-zero;
   (3) Qwen bridge route — confirmatory for protection/lesion/preference,
   substitution semantics development; (4) externalization — development,
   wedge unresolved, Bank-W pair underpowered (0.7788 at 16; first
   passing count 18); (5) transport premise model-dependent and gated —
   Gemma closed at tested scope, OLMo in-band failed at tested ε with a
   Think-only L56 late anchor; (6) operator convergence ≠ instrument
   invariance — methods, scoped Q-L4.
6. **Ledger rows cite registry events, never prose.**
