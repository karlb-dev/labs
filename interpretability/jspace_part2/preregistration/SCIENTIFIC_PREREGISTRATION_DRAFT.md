# SCIENTIFIC PREREGISTRATION — J-space Part 2 confirmatory campaign
## STATUS: DRAFT v1 (2026-07-28) — awaiting user review; becomes binding
## only when renamed SCIENTIFIC_PREREGISTRATION.md in a dedicated commit,
## BEFORE confirmatory item generation is revealed (addendum §9-0.3/0.4).

Written after the repair era passed its gates (G1 solver ✓, G2 lens
stability ✓, G3 intervention invariants ✓, G4 positive control ✓ — swap
0.76 vs 0.18; G0 provenance operational) and after 22 pilot evidence
items (registry). Pilot results inform hypotheses and margins; pilot data
will NOT be pooled into confirmatory cells.

## Primary hypotheses (frozen wording)

**HP1 (post-training reorganization).** Under the paper-faithful
output-protected dynamic J-ablation with full-sequence paired scoring,
the two-hop-vs-one-hop effect contrast (Δtwohop − Δonehop, both vs
baseline, hard-matched item sets) differs across the OLMo lineage
checkpoints (base, Think, 3.1-Instruct) and Qwen3.6-27B, ordered as
observed in pilot: base ≈ 0 < Think ≈ 0 < Instruct < Qwen — i.e. the
dissociation strengthens with instruct/hybrid post-training and is not
explained by measured occupancy (which is flat across the OLMo trio).

**HP2 (fact-accessibility channel).** Protected-ablation damage on
single-hop recall is predicted by fact accessibility (baseline logprob /
frequency), not task depth: easy-fact deltas < hard-fact deltas ≈ 0 on
the base model; this interaction replicates on ≥1 post-trained model.

**HP3 (indirect internal-content tail).** On think-trained OLMo, items
with answers inside the clean top-10 (protected) still show >1-nat
deletion at a rate exceeding the matched random control's, and the tail
membership reproduces across independent lenses (pilot: 15/17, corr .99).

**HP4 (capacity boundary).** Paper-defined occupancy and excess variance
for all tested open models fall below half the paper's Claude band
(occupancy < 5, excess < 3%), robust to lens fit size within the
validated stability envelope.

**HP5 (load engagement).** Static/persistence-selected ablation effects
emerge on high-load working-set items (C1 battery) where corpus-level
statics were null, OR equivalence within margins closes the load escape
hatch.

## Primary endpoints & analysis (binding)

Full-answer-sequence conditional logprob, paired per-item deltas,
family-clustered bootstrap + mixed model per addendum §12.2; binary
accuracy per §12.3; Holm correction over the FIVE contrasts above;
BH-FDR for prespecified secondaries; equivalence bounds for any null
claim. Deterministic teacher-forced primary; sampled secondary.
Chat-rendered cells use generation-based endpoints ONLY (pilot register
confound, evidence `a2a-mode-grid-qwen-v1`).

## SESOI margins (calibrated from pilot dispersion; final after power sim)

answer-seq lp: 0.5 nats · accuracy: 10pp · prose NLL: 0.10 nats/tok ·
occupancy: ±1 median unit · excess share: 1pp absolute.
Pilot paired-delta cluster SEs ran 0.15–0.30 nats at n=60/22 families →
n per confirmatory cell: simulate for 90% power at 0.5 nats with ICC from
pilot parquets; floor n=90 items across ≥30 independent families per
task (supersedes the habitual n=60; addendum §12.6).

## G6 POWER-SIM RESULT (2026-07-28, evidence `g6-power-sim-v2`) — DESIGN
## DECISION REQUIRED AT FREEZE (user)

The simulation (pilot-calibrated MoM components + family-block bootstrap,
Holm-worst alpha 0.01) shows the floor above is NOT attainable for mean
endpoints: protected-arm deltas are a zero-mode + heavy-tail mixture
(sig_e 1.2–1.5 nats within family, ICC ≈ 0.4), so the 0.5-nat mean-delta
primaries reach only ~24–42% power even at n=180/60 families, and no
affordable n reaches 90%. TOST equivalence at the same margin needs
n≈300. The TAIL-RATE endpoint (per-item >1-nat protected deletion,
J vs matched-random, paired, family-clustered — exactly the pilot's
"the tail is the phenomenon" observation) is well powered: 90%+ within
the extended grid at 10pp rate SESOI. Pilot tail rates and the full
grids are in `metrics/cross_model/g6_power_sim.json`. Cross-model
per-item delta correlations came out lineage-structured (Think–Instruct
0.687, OLMo–Qwen ≤0.14) — HP1's pairing advantage exists only within
the OLMo lineage.

Options for the freeze (pick one before confirmatory item generation):
(a) restate tail-carried primaries (HP3, and the HP2 accessibility
interaction) on the tail-rate endpoint at 10pp SESOI; keep mean deltas
for HP1's ordering contrast with the SESOI raised to the pilot-observed
0.8-nat scale OR HP1 demoted to estimation-with-CI (no binary test);
(b) keep 0.5-nat mean primaries and accept n≈150–200 families/task
(≈2–3× the compute envelope); (c) revise margins. Recommendation on
the science: (a).

## Item pools (partition-first)

Stage-3 bank: ≥30 templates/family across the ten C1 families + expanded
hard one-hop pool (pilot pool was 68/113 ceiling — new candidates target
lp ∈ [−9,−1] on Think) + paper released sets verbatim. Partitions
dev/pilot/confirmatory/replication hashed BEFORE any confirmatory
outcome is viewed. The 41-item dev set and all 22 pilot items stay
dev-tier forever.

## Exclusions, deviations, reruns

Per addendum §11.6; deviations logged in preregistration/DEVIATIONS.md;
independent rerun of the primary table required before release (§9-8.3).

## What the pilot may NOT do

Pilot cells (all 22 items in the registry) may inform design and margins
but are excluded from confirmatory statistics; no confirmatory claim may
cite a pilot number except as a preregistered prediction's origin.
