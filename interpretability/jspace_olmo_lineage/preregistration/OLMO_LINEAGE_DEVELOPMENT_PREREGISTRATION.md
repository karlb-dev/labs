# OLMo lineage development preregistration

Status: prospective development/methods plan, frozen before any OLMo Bank-W
baseline or intervention outcome is opened on this track.

## Boundary and graph

The scientific import boundary is Phase 4 commit `3b041735...`. The package
may read known Phase 3/4 development artifacts by exact hash. It may use only
the Bank-W development partition. It may not open untouched Phase 4
intervention outcomes or self-sign an independent-review or PI field.

The lineage graph is Base -> OLMo-3 32B Think -> OLMo-3.1 32B Think, with
OLMo-3.1 32B Instruct represented as a sibling endpoint under a distinct,
unobserved post-training recipe.

## Six axes

The study reports coordinate availability, sparse capacity, causal
utilization, downstream consumption, external-state substitution, and temporal
organization separately. It does not collapse them into a workspace score.

## Hypotheses

- H1: a broadly conserved thin dictionary is recruited or routed differently
  after post-training.
- H2: supplied or redundant external state substitutes for the internal
  channel on Think checkpoints.
- H3: Instruct routes state through output-adjacent or alternative receivers.
- H4: continuous answer commitment moderates causal cost.
- H5: one unobserved post-training stage installs the Think-path effect; this
  is stated-unresolvable if no genuine intermediate checkpoint is released.
- H6: per-checkpoint finite-dose transport validity may moderate the apparent
  lineage trajectory.

## O1 capability service obligation

Before lineage intervention science, run OLMo-3.1 32B Think and Instruct on
the exact Phase 4 Bank-W development capability protocol: 24 families, eight
seeds, derived/once, low and high load, 384 rows per model, full eight-answer
sequence-log-probability scoring. Each model passes only if both accuracies are
at least 0.70 and the family-bootstrap 90% interval for high-minus-low accuracy
lies wholly in `[-0.08, +0.08]`. Emit the early import bundle immediately.

## O2 capacity classification, fixed before Base is opened

Use one shared, ordered 120-prompt corpus with 30 prompts each from factual,
arithmetic, SQL/code, and neutral prose; globally center activations; measure
layers 24/32/40; use the paper-defined marginal-gain crossing against three
matched random dictionaries; and report occupancy, raw share, random share,
centered excess, solver diagnostics, strata, and prompt-bootstrap intervals.

Pairwise classification is:

- stable: absolute centered-excess difference <0.25 percentage points and
  median occupancy unchanged, with the equivalence interval inside that
  margin;
- small shift: 0.25--1.0 percentage points or one occupancy unit;
- material shift: >1 percentage point or >1 occupancy unit;
- unresolved: interval too wide.

Base is expected to fail many task-capability cells. Capacity does not require
task capability. Any Base O4 failure is reported as `gated_out`, never silently
omitted and never imputed as zero effect.

## O4 predictions, frozen before any intervention cell

The primary arm contrast is `specific = (LP_J-LP_0) - (LP_C-LP_0)`, where C
is exact instantaneous rank-and-energy matched control. Negative is greater
J-specific damage.

| account | high derived once | supplied | repeated | checkpoint pattern |
|---|---|---|---|---|
| external-state substitution | more negative on Think | attenuates toward zero | attenuates toward zero | visible by 3.0 Think; Base/Instruct nearer zero or differently organized |
| generic difficulty | J and matched control both worsen | no specific attenuation after margin adjustment | no specific attenuation after margin adjustment | shared across models |
| capacity | load slope follows measured capacity | no independent branch signature | no independent branch signature | ordered by capacity, not checkpoint role |
| output-adjacent | damage concentrates near commitment | may remain negative | may remain negative | supplied/repeated cells need not attenuate |

The named frozen contrasts are load engagement, supplied-state substitution,
redundancy substitution, and the load-by-externalization mechanism term.
Equal-family estimates, family bootstrap intervals, exact family sign flips,
leave-one-family-out sensitivity, candidate accuracy, and LP endpoints are
reported. Baseline covariates are sensitivities and cannot rescue a failed
primary pattern.

## Stop and scope rules

The O1 -> O2 -> O4 spine is mandatory. O3 audits existing lens provenance
before any refit. O5 is a minimal crossed activation/lens/readout design. Axis
D is capped at one clean within-model rescue demonstration and half a GPU day.
Intermediate-checkpoint work begins with inventory only. No controlled
training starts in this release block. The first release stops at a reviewed
state-of-record and Phase 4 import bundle.

## Prohibited claims

The track may not claim that Think training creates a global workspace, that
the reasoning objective is the causal variable, that Instruct lacks a
verbalizable channel, that capacity is unchanged from pretraining before Base
is measured, that dictionaries are identical from common-frame stability, or
that a single patched receiver is the workspace.
