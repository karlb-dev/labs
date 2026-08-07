# PAPER_ROUTE_DECISION.md — P6

**Decision: Route B — an empirical causal-channel paper (Paper A) plus a
methods/applicability companion (Paper B).** Made after P0–P5 under the
frozen §9.5 rubric, with the reconstruction audit green (286 targets, 0
frozen-number failures). Route B was the addendum's working default;
the analysis below is what *earns* it. **Status: analysis decision,
pending PI ratification** (recorded per campaign practice; nothing
publishes before the P9 audit passes and the PI signs).

## Rubric scoring (A = one integrated paper, B = two papers)

| Criterion | Route A | Route B | Evidence |
|---|---|---|---|
| One-sentence thesis | fails: "channels exist AND instruments have validity boundaries" is two theses | each paper has one sentence (below) | A6 matrix: the transport/fit-invariance columns are orthogonal to the causal columns; Qwen was never transport-gated at all |
| Highest-tier evidence strength | confirmatory core diluted by methods volume | A leads with the campaign's only confirmatory+replicated cell; B leads with Q-L4 | A6: exactly one confirmatory+replicated cell exists |
| Tier transitions in main text | ≥5 (conf → dev → methods → dev → methods) | A: 2 (conf anchor → dev OLMo → boundaries); B: 0 (all methods) | claim ledger tiers C1–C7 |
| Figure economy | 10–12 main figures forced | 6 + 6 (E1–E6, M1–M6 ledgers) | P7 ledgers below |
| Reviewer-objection closure | the H6/Q-L4 objections dominate an empirical narrative | A cites the A5 reconciliation; B owns the boundary as its thesis | `A5_H6_PHASE3_RECONCILIATION.md` pre-built |
| Overclaim risk | high: methods caveats read as hedging, empirical claims read as surviving despite them | low: each claim sits at its native tier in its native paper | EVIDENCE_TIER_RULES rule 4 |
| Reproducibility | one giant package | two scoped packages sharing one verified data release | A8 durability table |
| Submission readiness | months of restructuring | A is seeded (`kburtram_jspace.tex`, 30/31 numbers verified); B assembles from frozen syntheses | recon audit |
| Need for new GPU data | none either way | none | CPU-only phase |

Route B wins every row it differs on. The §9.1 risk for Route A — "a
sprawling manuscript that appears to change thesis every six pages" —
is precisely what the completed record would produce.

## The two theses (one sentence each)

- **Paper A:** Open ~30B models contain verbalizable, causally
  load-bearing knowledge-access channels whose effects are carried by
  direction content rather than dose, whose use is bridge-mediated on
  Qwen, and whose organization varies with training checkpoint on the
  OLMo lineage — with all claims at their registered tiers.
- **Paper B:** An averaged Jacobian operator can converge strongly and
  stably in its aggregates while its sparse selections, mechanism
  endpoints, finite-dose tangent predictions, and even its backward
  semantics under version-pinning fail invariance — so J-lens causal
  work needs a per-checkpoint, per-dose, per-backend preflight, which
  we specify.

Paper B's spine (per addendum §2.1): the A120→A1000 convergence split
(Q-L4), selection-margin/influence falsifiers, Gemma's calibrated
finite-scale mismatch, the OLMo H6 boundary, the transport gate — plus
the **runtime-identity incident as a named section** (C7).

## Dispositions

1. **OLMo standalone paper: no.** Disposition = *empirical-paper
   section plus technical report*. The lineage carries Paper A §5 at
   development tier; `olmo_lineage_parallel_phase.tex` (already at
   Study-2 terminal state) is re-labeled a technical report /
   appendix source. A standalone submission would stack four gates
   (wedge capability-gated, H6 in-band fail, Bank-W unpowered,
   natural-experiment boundary) against a development-tier core —
   §9.3's warning applies: 16 existing pages are not a reason.
2. **Gemma handout: remains educational**, case conclusion updated to
   the Study-2 V2 calibrated license (P8, §11.5); Paper B cites it
   rather than absorbing its derivations.
3. **Phase 4 handout:** regenerate to Q-L4 Terminal C with the Study-2
   import boundary (P8, §11.6) — permitted now that the route decision
   exists.
4. **`Mapping_Neural_Curvature.pdf`: retired as a paper seed**
   (NotebookLM-generated deck, no source, pre-Study-2 framing; see
   `MAPPING_NEURAL_CURVATURE_INVENTORY.md`). Excluded from the
   claim-bearing release; any teaching reuse requires a Study-2
   consistency check.
5. **`kburtram_jspace.tex` is Paper A's working draft** (fresh start,
   evidence-id native). P8 applies the four register corrections (U1–U4)
   and the A2 wording boundaries.

## Public-artifact sequencing (addendum §2.5 — teed up for the PI)

Recommended: (1) Alignment Forum / LessWrong post built from the claim
ledger and survival timeline — "we replicated, repaired the
instruments, and here is what survived"; (2) arXiv Paper A; (3) Paper B
with the transport-gate/preflight protocol release. Nothing publishes
before P9 passes and the PI signs.

## Inclusion rules now in force (from §9.2)

- Paper A admits only: the confirmatory/replication anchor (C1), bridge
  mechanism (C3), OLMo development triangulation (C2), the boundaries
  needed to interpret them (C4–C6 summarized, A5 reconciliation cited),
  and limitations. Detailed instrument autopsies go to B or appendices.
- Paper B admits only: instrument-validity results (C5–C7, Q-L4 detail,
  falsifiers, calibration protocol, preflight standard). No empirical
  causal claim is re-argued there; it cites A.
- One master evidence/figure ledger serves both; cross-reference, never
  duplicate.
