# 1 · The problem

Working title (per nextsteps §18.1): **Verbalizable Causal Channels Are
Model- and Task-Dependent in Open Language Models.**

## 1.1 The claim under test

The July-2026 paper "Verbalizable Representations Form a Global
Workspace in Language Models" (transformer-circuits.pub/2026/workspace)
proposes that a small, verbalizable subspace of the residual stream —
read out by a fitted Jacobian lens — forms a *global workspace*: sparse,
broadcast, and causally privileged, such that deleting its content
selectively destroys multi-step ("composed") behavior while sparing
direct recall. The claim matters because it is the strongest
mechanistic-interpretability result to date connecting a *readable*
representation to a *causally load-bearing* one, and because "workspace"
imports a functional architecture — a shared bus between specialist
processes — that most circuits-level results do not claim.

## 1.2 Why replication is not a formality here

The paper's causal instrument is a *dynamic, output-protected ablation*:
at every position, the top-k token-row directions selected by the J-lens
are removed from the residual stream, except that rows corresponding to
the model's current top-p output candidates are protected from
selection. Every step of that sentence hides a failure mode we
encountered and had to control:

1. **Output deletion masquerading as content deletion.** Unprotected
   live ablation mostly deletes the *emerging output*, not internal
   content — behavioral damage is then trivial and universal
   [established after Phase 2; nextsteps §18.5].
2. **Dose confounds.** The J arm removes a particular *amount* of
   activation energy at a particular effective rank per position. Any
   control that removes less, elsewhere, or incoherently, makes the J
   arm look special for free. The primary control must match achieved
   rank *and* removed energy, instantaneously, per position
   [`mc-dev-validation-olmo31-think-v2`].
3. **Selection-vs-geometry.** Protecting output *token IDs* from
   selection does not make the removed span *orthogonal* to the output
   directions. This asymmetry — the paper's protection contract
   protects labels, not the span — was flagged at Phase 2 close
   [`p2-label-vs-span-protection-openq-v1`] and is the opening question
   of Phase 3 (§2.3 of the Phase 3 plan; development-tier measurements
   now exist and motivate a span-safe primary arm).
4. **Instrument-unit hazards.** The assay is sensitive to tokenizer BOS
   conventions and concatenation conventions at the ~1.5-nat level per
   item — larger than most effects under study
   [`AMENDMENT_1_BOS_UNITS`].

Separating "the model lost the fact" from "the model lost the ability
to say anything" from "the model was injured proportionally to how much
of anything was removed" is the entire technical problem.

## 1.3 What a replication can and cannot conclude

The Phase 2 campaign preregistered two primary hypotheses on open
models (OLMo 3.1 32B Think/Instruct as the lineage primaries, Qwen 3.6
27B as the mode/architecture contrast), with the Claude-family paper
results as the anchor being tested for *transfer*:

- **HP1** (task-shape): composed (two-hop) items lose more than direct
  (one-hop) items under protected J ablation, model-dependently.
- **HP3** (specificity): the per-item damage distribution has a
  J-specific heavy tail that an exact rank-and-energy matched control
  does not reproduce.

Under §4.5 of the governing plan, no outcome of these tests supports
claims about consciousness, about a model "lacking a workspace," or
about label protection preserving the output subspace. The calibrated
summary sentence the evidence currently supports is (§0.2, verbatim):

> Qwen exhibits a paper-shaped protected-ablation dissociation under
> the Phase 2 assay, while OLMo exhibits a different,
> accessibility-dominated lesion response.

The stronger sentence — that the full workspace causal signature has
been robustly replicated on Qwen and refuted on OLMo — is explicitly
*not yet licensed*; the four blocking gaps (thin two-hop leg, span
geometry, the missing prose exact control, and full clean-room
reproduction) define Phase 3's agenda.
