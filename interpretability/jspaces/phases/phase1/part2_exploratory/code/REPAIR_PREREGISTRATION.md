# REPAIR PREREGISTRATION — Workstream R (assay repair and conformance)

Adopted 2026-07-27 following the forensic review
(`jspace_part2_plan1_addendum.md`, versioned alongside this file). This is
the FIRST of two preregistrations (addendum §9 Stage 0.3): it governs the
assay-repair phase. The SCIENTIFIC preregistration (model/task hypotheses,
final margins, sample sizes from power simulation, confirmatory holdout
freeze) is written only after the gates below pass — deliberately, so
debugging cannot contaminate confirmatory claims.

Supersedes `preregistration.md` (2026-07-27 morning) for everything
confirmatory. That file remains the honest record of what was committed
before any Part-2 data; its A0/B1 decision rules are downgraded to
exploratory heuristics by addendum §7.3/§18.4/§18.8.

## Standing reclassification (addendum §0, §2, §6)

- All Part-1 results and all Part-2 cells produced BEFORE the R-gates pass
  are **exploratory** (evidence vocabulary: established / provisional /
  exploratory / not-currently-identified — addendum §1.4).
- Part-1 headline wording is corrected in `REPORT_v2_ERRATA.md` (repo,
  beside REPORT_v2.md). SL1 ledger wording is frozen-historical; nothing
  inherits it without Stage-2 re-audit under the repaired assay.
- Already-banked Part-2 cells relabeled: A0 = lens-transfer geometry
  experiment (exploratory; multiple-opportunity rank caveat, addendum
  §5.3.1); D-traces = output-alignment anchor (exploratory; renamed from
  "occupancy" — that term now reserved for the paper's capacity estimand).

## The repair workstream (R0–R7) and its validation tests

**R0 Provenance freeze.** `run_manifest.json` at study root: code commit,
dirty flag, model id+resolved revision+config/tokenizer SHA-256, lens
hashes + jlens commit, prompt-set hashes, environment lock hash, GPU.
Result files must reference the manifest + their own input hashes; cached
artifacts are refused when any declared input hash differs. Artifact
inventory of every referenced v1/v2 artifact (path, size, SHA-256,
producing script, complete/partial/selected status, per-item availability,
confirmatory-reuse suitability). Missing raw data is declared, never
synthesized from aggregates.

**R1 Paper-faithful output-protected dynamic ablation.** Clean pass →
cache top-10 clean output token ids per scored position → paper-consistent
non-negative active-direction score → EXCLUDE protected output-token
directions → rank-safe projector from the remaining top-10 → rerun.
Tests: no protected id ever selected; synthetic intended-output direction
untouched; k shrinks when too few valid positive directions exist;
duplicate rows don't inflate effective rank; fp16/fp32 selection agreement;
determinism under fixed inputs.

**R2 Paper-defined occupancy + excess variance.** Marginal-gain crossing
vs equal-size random dictionaries (Delta_J(K) vs Delta_R(K) distribution;
preregistered crossing rule), excess variance at median occupancy with
GLOBALLY centered mergeable moments. Solver validation on synthetic
mixtures with known support (recovery, monotonicity, duplicates,
high-coherence, half-precision, determinism) against NN-OMP + NNLS-on-
support + high-accuracy reference. THEN recompute OLMo and Qwen Part-1
capacity values with the one final algorithm (old-vs-new errata figure).

**R3 Rank-safe frozen intervention + geometry-matched control family.**
SVD/rank-revealing bases everywhere raw QR was used; report requested k
AND achieved effective rank; controls matched on effective rank. Control
family per item/layer (addendum §5.7): J-selected, isotropic random
(pool-size matched, same rule), J-rotated (spectrum+Gram preserving),
label-shuffled J, logit dictionary, matched non-J (energy+rank+spectrum),
counterfactual content, distractor content. Per-projector logging:
selected ids+strings, scores+signs, singular values, numerical rank,
removed energy (prompt/answer position/decode), clean-top-10 overlap,
projector hash.

**R4 Phase, scoring, and statistics.** (a) Phase-resolved hooks
(prefill-only / decode-only / both / neither) with sentinel tests: prefill
hooks never fire on decode, decode hooks never alter cached prompt KV,
fire-count assertions per item, exception-safe detach. (b) Full-answer-
sequence conditional logprob (per-token logprobs, sum, mean; prespecified
alias aggregation); first-token retained as diagnostic only. (c) Paired
per-item records (parquet + human-readable sample) with the addendum §5.12
schema; locked analysis: paired deltas, cluster bootstrap by true
generation unit, equivalence-from-interval, small Holm-corrected primary
family; "CI-overlap ⇒ null" and separate per-condition bootstraps are
retired.

**R5 Released paper tasks.** Implement the released experiment sets
(probe-swap, selectivity, capacity, dual-task at minimum; the rest
enumerated in addendum §7.5-C2) with model-specific tokenization audits
and frozen adaptation rules recorded before outcomes are inspected.

**R6 Tiny-model golden tests.** Full pipeline on a small decoder: lens
fit/save/load; manifest hash reproduction; phase hooks; protected tokens
untouched; nested doses nested; per-item records regenerate aggregates;
interrupted descriptive == uninterrupted covariance; figures pure
functions of raw records.

**R7 Stage-2 re-audit of Part 1.** With R1–R6 passed: protected vs
unprotected dynamic grid (+ matched dynamic random + dynamic J-rotated) on
dev/pilot sets with the paper's multihop/selectivity tasks; capacity
recompute; frozen control family on a development set; CoT endpoint repair
(all traces saved, immutable projectors, post-`</think>` final-answer
primary endpoint, balanced direct-vs-think factorial).

## Gates (no model matrix before these pass)

| gate | pass condition |
|---|---|
| G0 provenance | all inputs pinned + hashable; manifest enforced |
| G1 solver | synthetic recovery + batching/resume invariants pass |
| G2 lens | hidden validation + negative-control lenses (label-shuffled, layer-shuffled) pass; recipient ≥ transfer |
| G3 intervention | protected ids untouched; phase hooks correct; rank+energy logged |
| G4 positive control | released swap/ablation effect appears without general damage |
| G5 task | baseline capability + shortcut audits pass |
| G6 pilot precision | design resolves the SESOI/equivalence margins |

## Planning-default equivalence margins (provisional; final values set by
pilot calibration + power simulation in the scientific preregistration)

Answer-sequence logprob 0.5 nats · accuracy 10 pp · prose NLL 0.10
nats/token · pretraining top-1 agreement 5 pp · occupancy 20% relative.

## Stage-2 decision gate (verbatim commitment, addendum §9)

Protected dynamic ablation selective → Part 2 = characterization +
post-training localization of a positive. Null with strong positive
controls + equivalence → rigorous boundary-of-generalization study.
Positive controls fail / solver-lens instability dominates → methods paper
on J-lens reliability; model breadth pauses.

## Primary-pair pin update (addendum §7.3, hub-verified 2026-07-27)

Primary lineage endpoints: `allenai/Olmo-3.1-32B-Think` +
`allenai/Olmo-3.1-32B-Instruct`; anchor `allenai/Olmo-3-1125-32B` (base).
Documented graph: base → Think-SFT → Think-DPO → {Olmo-3-32B-Think,
Olmo-3.1-32B-Think (base_model: Olmo-3-32B-Think-DPO)}; base →
Instruct-SFT → Instruct-DPO → Olmo-3.1-32B-Instruct. No 3.1-Think
SFT/DPO exist; intermediates are the 3.0 ones. Part-1's Olmo-3-32B-Think
= historical replication cell + lineage member, not the sole Think
endpoint. Matched-lineage NATURAL experiment, not one-variable control
(wording per addendum §7.3). Qwen: A2a = same-checkpoint official
thinking on/off via chat template (raw completions are NOT the official
non-thinking mode); A2b = separate lineage checkpoint if compatible.

## Interim-work rules until gates pass

- The running 120-prompt Olmo-3.1-32B-Instruct lens fit continues
  (recipient fits are REQUIRED; 120 becomes the first nested point of the
  B1 curve; canonical n chosen per addendum §5.2 equivalence-to-n=500).
- GPU cells executed before gates (B3 frozen-logit pilot, own-lens
  re-gate) carry `"evidence_tier": "exploratory-pilot"` in their JSONs and
  feed instrument design only.
- Every figure/table/report row generated before gates is labeled
  exploratory. The matrix master dataset carries a `tier` column.
