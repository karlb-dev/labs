# SCIENTIFIC PREREGISTRATION — J-space Part 2 confirmatory campaign
## STATUS: DRAFT_V2_PRE_REVIEW (frozen as historical record 2026-07-28;
## N0 snapshot `reports/PILOT_SNAPSHOT_VM7.json`, tag
## `jspace-part2-pilot-vm7`). SUPERSEDED IN PLACE by the review cycle:
## PI decisions D1–D7 are resolved in
## `reviews/jspace_lab_nextsteps_2_2_addendum.md` §3, and the successor
## document is `SCIENTIFIC_PREREGISTRATION_CANDIDATE.md` (stage N5).
## This draft stays for the record; do not edit further.
## Binding rule unchanged: nothing confirmatory may run until a candidate
## is renamed SCIENTIFIC_PREREGISTRATION.md in a dedicated commit.

Written after the repair era passed its gates (G0 operational, G1 solver
✓, G2 lens stability ✓, G3 intervention invariants ✓, G4 positive control
✓ — swap 0.76 vs 0.18, G6 power ✓) and after 45 live pilot/dev evidence
items (7 withdrawn; see registry). Pilot results inform hypotheses and
margins; pilot data will NOT be pooled into confirmatory cells.

---

# ⚠ DECISIONS REQUIRED BEFORE FREEZE — read this section first

Six decisions are yours. Each blocks the freeze; nothing downstream can
run until they are settled. D1 and D2 are the consequential ones.

**D1 · PRIMARY ENDPOINT (blocks everything).** G6 (`g6-power-sim-v2`)
found the drafted 0.5-nat MEAN primaries structurally underpowered:
protected deltas are a zero-mode + heavy-tail mixture (within-family σ
1.2–1.5 nats, ICC ≈ 0.40), giving only 24–42% power even at n=180 across
60 families; no affordable n reaches 90%, and TOST needs n≈300. The
TAIL-RATE endpoint is well powered at a 10pp margin. **But switching is
not free** (`tailrate-endpoint-crossmodel-v2`): the two endpoints do NOT
rank the models identically — mean order is base < Think < Instruct <
Qwen, rate order is base < Instruct < Think < Qwen. Think and Instruct
exchange places, so this changes HP1's claim rather than relabeling it.
Options: **(a)** binary primaries on the tail-rate endpoint + HP1 as
mean-endpoint estimation-with-CI, no binary test *[recommended]*;
**(b)** keep 0.5-nat mean primaries and accept n≈150–200 families/task
(2–3× the compute envelope); **(c)** revise the margins.

**D2 · WHICH LENS DO CONFIRMATORY CAUSAL CELLS USE? (new, not in v1 —
and it may matter more than D1).** Today's linearity work
(`local-linearity-v3-*`, `linearization-faithfulness-*-v2`) found that
OLMo's source→target transport IS linear across the paper's band (scale
ratio 1.98–2.02 at ε=0.1–0.2), yet the fitted campaign lens predicts the
true response with only ~0.49 cosine at L24 and captures 41% of its
magnitude. Since the map is linear, that gap is **estimation error** —
jlens averages the Jacobian over positions (`grad.mean(dim=1)`) and over
fitting prompts. This is the campaign's H7 mean-J mismatch, now measured.
**Implication: every J-direction causal claim in this campaign (and in
the paper) may rest on a needlessly degraded estimator.** The direct test
is cheap and NOT yet run (task #12): compute a true per-position Jacobian
by autograd and compare its faithfulness against the averaged lens. If
per-position is markedly better, confirmatory causal cells should
arguably use per-context Jacobians, and the whole confirmatory design
changes. **Recommendation: run task #12 BEFORE freezing.**

**D3 · IS GEMMA IN THE CONFIRMATORY MATRIX?** A3 is complete at pilot
tier and the answer is a premise failure, not a result: Gemma-4's
transport is nonlinear at every layer and scale (`local-linearity-v3-gemma4-31b`),
so no Jacobian models it however well estimated — which is why its fitted
lens is identified-but-useless and reads worse than a plain logit lens.
Options: **(a)** keep Gemma as a reported METHODS boundary case, run no
confirmatory J cells on it *[recommended — the honest finding is already
banked]*; **(b)** drop it entirely; **(c)** spend GPU on a
nonlinearity-tolerant variant (no such instrument exists today).

**D4 · TAIL THRESHOLD.** If D1 lands on the rate endpoint, the >1-nat
threshold must be frozen. Pilot used −1.0 nat; threshold sensitivity is
reported in `g6_power_sim.json` and `tailrate_endpoint.json` so the
choice is made with its arbitrariness visible. Any later change is a
logged deviation.

**D5 · ITEM PARTITION.** The Stage-3 bank now MEETS the floor
(`c3-bank-status-v1`: 126 hard items across 34 canonical families vs
n≥90 / ≥30 required). It is deliberately NOT partitioned — splitting into
confirmatory/replication and hashing the partition is a freeze action,
and must happen before any outcome on those items is viewed. The frozen
41-item dev set and all pilot items stay dev-tier forever.

**D6 · G5 IS NOT RUN.** The task-gate (baseline capability + shortcut
audits) is the one R-gate never executed. Either run it before freeze or
record an explicit waiver with reasoning.

---

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

**Reality check on option (a)** (evidence `tailrate-endpoint-crossmodel-v2`,
which computes the proposed endpoint on the existing pilot grids rather
than trusting the simulation): at the pilot n=60 the J-vs-matched-random
tail-rate difference excludes zero on **only 1 of 4 models** (Qwen,
+0.267 [+0.140, +0.489]); base +0.067, Instruct +0.100, Think +0.133 all
straddle. That is consistent with G6 — the endpoint needs the larger n,
it is not free at pilot size. **More important: the two endpoints do NOT
rank the models identically.** Under the mean endpoint the ordering is
base < Think < Instruct < Qwen; under the rate endpoint it is base <
Instruct < Think < Qwen — Think and Instruct exchange places. So
adopting the rate endpoint is a change of scientific claim, not a
relabeling: **HP1's ordering must be restated against whichever endpoint
is frozen, and the pilot ordering that motivated HP1 is endpoint-
dependent.** A defensible resolution is to keep HP1 as a
mean-endpoint estimation-with-CI claim (no binary test, no power
requirement) and put the binary primaries on the rate endpoint, so each
claim is made where it is identified. That choice is the user's.

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
