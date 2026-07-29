# SCIENTIFIC PREREGISTRATION — CANDIDATE
## J-space Part 2 confirmatory campaign

**STATUS: CANDIDATE. NOT BINDING.** It becomes binding only when renamed
`SCIENTIFIC_PREREGISTRATION.md` in a dedicated freeze commit after
explicit principal-investigator sign-off (nextsteps_2_2 §7-N5). Until
then: no confirmatory cell, no partition generation, no outcome viewed on
a confirmatory item.

Supersedes `SCIENTIFIC_PREREGISTRATION_DRAFT.md` (closed as
`DRAFT_V2_PRE_REVIEW`). Governing documents: `../reviews/jspace_lab_nextsteps_2_2.md`
(repair spec) and `../reviews/jspace_lab_nextsteps_2_2_addendum.md` §3
(PI decisions D1–D7, which govern on conflict).

Every design number below comes from a **post-repair** evidence id. The
pre-repair numbers are not used, because the family field they clustered
on was invalid.

---

# 0 · What changed since the draft, and why it matters

Four repairs moved the design, not just the code.

**0.1 The clustering unit was wrong, and it was load-bearing.**
`battery.py` derived the statistical family as `name.split("-")[0]`. Under
the audited map (`data/probe_swap_family_map.json`, 120 items hand-mapped
by the rule *(second-hop relation, answer type)*) the pilot's 60 two-hop
items are **25 canonical families, not 38 raw labels**, and one family
(`country_capital`) holds 11 of them. Every published interval, ICC and
the whole power simulation inherited the error.
`n2-corrected-family-pilot-v1` recomputes all of them.

Point estimates are unchanged — clustering affects uncertainty, not means
— but:

- **ICC was never uniform.** The defective field made all four models look
  alike at 0.37–0.42. Audited: base **0.66**, Think **0.56**, Instruct
  **0.17**, Qwen **0.75**.
- **One conclusion flips.** Instruct one-hop moves from
  −0.45 [−1.03, **+0.06**] to −0.45 [−1.29, **−0.01**]. The VM6 ladder
  called Instruct a "weak forward dissociation" because its one-hop
  interval straddled zero; corrected, it marginally does not. It is
  knife-edge and it reverses again under family weighting, which is why
  §6.3 fixes the weighting estimand in advance.
- **Think and Instruct still exchange places between endpoints**, so the
  D1 disclosure survives the repair.

**0.2 The binary primary is not affordable on the OLMo pair.**
`g6-mde-v1` inverts the power question: given a bank of *m* families, what
is the smallest effect detectable at 90%? At **60 canonical families** —
the bank we actually have:

| model | MDE rate @90% | pilot gap | MDE nats @90% | pilot mean |
|---|---|---|---|---|
| OLMo Think | 0.30 | 0.133 | 1.00 | −0.52 |
| OLMo Instruct | 0.30 | 0.100 | 0.75 | **−0.80** |
| OLMo base | 0.20 | 0.067 | 0.75 | −0.14 |
| Qwen | **0.25** | **0.267** | 1.50 | −0.91 |

Only **two** cells can carry a binary test: **Qwen on the rate endpoint**
and **Instruct on the mean endpoint**. The two endpoints have *opposite*
feasibility profiles across models. HP3's Think tail effect would need
>150 families to test as a binary. §3 is written accordingly: most
hypotheses are stated as estimation with intervals, and only where power
exists is a test declared.

**0.3 Context-conditioning does not rescue the Jacobian (D2 resolved).**
`h7-synthesis-olmo3-think-v1`. The decisive design point: for a *uniform*
perturbation the per-position and corpus-averaged estimators are
algebraically identical (`sum_s J_s @ d == P·(mean_s J_s) @ d`), so the
earlier faithfulness test could not see position-averaging loss at all.
Perturbing one position at a time — what the dynamic ablation does —
separates them. Result: position-conditioning does **not** improve
response direction at any band layer (−0.05, −0.04, −0.02), though it
roughly halves the magnitude error. Cell by cell, cosine tracks the
ground truth's *own* local linearity (within-layer Pearson r 0.76–0.90,
n=144), so the estimator is not the binding constraint — the ceiling is.
**The `contextJ_methods` arm therefore FAILS its committed dev gate and is
not admitted.** `meanJ_paper` is the sole primary lens.

Mitigating and worth stating plainly: along the lens's own top rows —
the directions the ablation actually selects — in-band cosine is **0.58**
versus **0.33** for random probes. The frequently quoted "~0.2 in band" is
a random-probe number and overstates the problem for the intervention
performed here.

**0.4 The capacity number was mislabelled and is now larger.**
`r2-occupancy-think-v2`. v1 computed a centered activation and never used
it; both shares were raw-energy while the prose called them centered
excess variance. Occupancy is unchanged (median **2** at L24/32/40 — it
never depended on the share definition). The corrected centered excess is
0.49% / 1.02% / **1.27% [1.18, 1.34]** versus raw 0.03% / 0.54% / 0.80%.
The capacity headline survives in slightly stronger form: still an order
of magnitude below the 6–10% reported for Claude.

---

# 1 · Scope, models and instruments

**Primary pair.** `allenai/Olmo-3.1-32B-Think` vs
`allenai/Olmo-3.1-32B-Instruct`, each with its OWN fitted lens. The pilot
Think cells used `Olmo-3-32B-Think` (revision `ebd033e4…`); that
checkpoint is a **lineage moderator, not the primary**, and no pilot cell
may stand in for a primary cell.

**External validation.** `Qwen/Qwen3.6-27B`, raw-completion protocol
frozen identically.

**Lineage moderators, after the primary pair only.** OLMo base-1125,
`Olmo-3-32B-Think`.

**Excluded from confirmatory causal work (D3).** Gemma-4-31B remains a
methods boundary case: fit identification, depth-opacity, finite-scale
transport and the norm-convention lesson are retained; no confirmatory
J-direction causal cell is run on it. Wording ceiling: *"the paper's
method as implemented does not transfer to this architecture"*, never
"Gemma has no workspace".

**Instrument.** `protected_dynamic_v2` (per-position prefill protection,
row-wise dose, phase control, structured per-position logs). The v1
module is retained only for historical reproduction. Any generation-based
cell uses the v2 path; the v1 generation path is disqualified because it
broadcast one final-position protection set across every prompt position.

**Lens.** `meanJ_paper` — the corpus-averaged Jacobian, exactly as the
paper defines it. Its measured cost is stated, not hidden: in-band
direction cosine ≈ 0.58 along selected rows, magnitude under-predicted
~2×, and a linearity ceiling no better estimator can pass. **That bound
applies to every J-direction causal claim in this literature, this
campaign's included.**

---

# 2 · Item bank and partition (D5)

**Bank.** `g5-item-manifest-v3`: 1052 items, of which **64 canonical
families hold ≥3 items inside the difficulty window** on the anchor model
(325 items). Composition: released probe-swap two-hop (90), the package
one-hop battery (30), and the Stage-3 hard one-hop bank (v2–v6).

**Clustering.** `canonical_family` is the **primary** unit.
`relation_group` — a coarser pooling (`work_to_creator`,
`entity_to_category`, `ordinal_superlative`, `structured_count`,
`process_name`, `definition_to_term`) — is a **prespecified sensitivity**.
Reporting both is deliberate: a reviewer who thinks the pooling should be
coarser reads that analysis rather than doubting the primary.

**Exclusions, applied before any outcome.** 15 items excluded for
answer-in-prompt leakage (e.g. *"The gland that secretes parathyroid
hormone is the"* → *parathyroid*), 4 duplicate prompts dropped across pool
versions. Zero bridge leakage; every two-hop item carries its swap
counterfactual.

**Partition.** `partition.py`, algorithm `family_split_v1`, split over
FAMILIES never items, greedy largest-first allocation balancing
difficulty tercile and answer-token length. It **refuses to run** without
`freeze_authorised=True`. The manifest will record seed, source-bank
hash, per-side family and item lists, stratum counts and a manifest
SHA-256. Generation happens **inside the freeze commit and nowhere else**.

**Open condition, stated rather than hidden.** Cross-model capability
cohorts (`lineage_anchor`, `cross_model_intersection`, `model_specific`)
require every confirmatory checkpoint's weights; only the anchor is
scored. **Completing them is a precondition of the freeze**, because a
bank selected on one model's difficulty smuggles a sampling bias into
every cross-model claim.

---

# 3 · Hypotheses

Each states its endpoint, its estimand, and — explicitly — whether it is a
**test** or an **estimate**. Where §0.2 shows a test is not powered, it is
declared an estimate. That is the honest form, not a hedge.

### HP1 · Post-training changes the protected-ablation task contrast
> Under the paper-faithful averaged-J output-protected dynamic ablation,
> the contrast between two-hop and difficulty-matched one-hop effects
> differs across `Olmo-3.1-32B-Think` and `Olmo-3.1-32B-Instruct`. Qwen3.6
> provides an external validation estimate. The claim is a model-by-task
> interaction plus the prespecified Think-vs-Instruct contrast — **not** a
> four-model rank order.

Endpoint: continuous full-sequence answer delta. **ESTIMATE with CI**
(mean-endpoint MDE 0.75–1.00 nats at 60 families vs pilot effects
0.52–0.80: a test is borderline at best). Tail-rate reported in parallel,
with the Think/Instruct endpoint swap disclosed. The pilot ladder is the
hypothesis's origin, not something confirmation must reproduce.

### HP2 · Accessibility, not nominal hop count, predicts one-hop damage
> Within each model, protected-ablation damage on one-hop factual recall
> varies with preregistered baseline-accessibility strata.

Accessibility = baseline full-sequence logprob (frozen, pre-outcome).
**ESTIMATE with CI**, per model. This supersedes the pilot's "easy-fact
effect" reading, which the ceiling check found by accident.

### HP3 · A protected internal-content tail exists on think-trained OLMo
> On the untouched Think confirmatory partition, the rate of items losing
> more than 1.0 nat under protected averaged-J ablation exceeds the rate
> under the primary matched control.

**Primary tail is stratified to protected-answer items only** (clean
answer rank ≤ protect_k) per the PI amendment: 21% of pilot tail items
were protection failures — items whose answers were unprotectable — and an
unconditional tail conflates indirect J-content damage with instrument
reach. All-items tail is prespecified sensitivity.
**ESTIMATE with CI** on OLMo (MDE 0.30 vs pilot gap 0.133); **TEST** on
Qwen, where power exists (MDE 0.25 vs gap 0.267).

### HP4 · Open-model occupancy lies below the reported Claude range
> Under the finalised solver and the **centered R²** excess definition,
> tested models have median occupancy and centered excess variance below
> preregistered boundary values derived from the reported Claude range.

**Scoped (PI amendment) to models satisfying the J-lens validity premises**
— linear transport at intervention scales and cross-corpus identification.
Gemma is excluded by premise failure, not counted as a below-boundary
data point. Report occupancy distribution by layer, right-censoring,
centered excess with CI, random-dictionary seed sensitivity, solver
sensitivity, crossing-rule persistence sensitivity (1/2/3), and fit-size
sensitivity. **ESTIMATE with CI plus a one-sided boundary comparison.**

### HP5 · High working-set load reveals or closes a static-span effect
> On synthetic and released high-load tasks passing G5, the
> ablation-by-load interaction is either directionally positive beyond
> the SESOI or statistically equivalent to zero within the frozen bound.

Separate multiplicity family B. Equivalence by proper TOST (90% interval
at α=0.05), never a 95% interval read as equivalence.

### HM1 · Context-specific transport (methods) — **RESOLVED, NOT ADMITTED**
Pre-run gate: median response-cosine improvement ≥0.20 over mean-J and
≥0.80 absolute on ≥2 band layers. Measured: **−0.04** median band
improvement, **0** band layers at 0.80. The arm is not admitted. Recorded
here because a preregistration should show the arms it rejected and why.

---

# 4 · Conditions

| condition | role |
|---|---|
| `baseline` | clean, no intervention |
| `meanJ_protected` | **primary** — paper-faithful averaged-J, output-protected |
| `primary_matched_control` | **primary control** — see §5 |
| `dynR_mechanics_control` | isotropic random dictionary, same machinery |
| `meanJ_unprotected` | diagnostic only, never primary |
| `logit_protected` | prespecified secondary (output-aligned geometry) |
| `contextJ_protected` | **not run** — HM1 failed its gate |

---

# 5 · The primary control must be geometry-matched

The pilot's random-dictionary arm is a **mechanics** control, not a
matched one. It is renamed `dynR_mechanics_control` throughout, and the
report may not call it matched. Naming discipline, fixed here:

`dynR_mechanics_control` · `dynJ_label_shuffled` · `dynJ_rotated` ·
`dyn_spectrum_matched_nonJ` · `dyn_energy_rank_matched_random`.

**RESOLVED (2026-07-29): the primary control is
`dyn_energy_rank_matched_random`** (`matched_control.py`). Per (item,
layer, position) it removes a RANDOM subspace matched **exactly, by
construction** to the J arm's achieved geometry at that site:

- same effective rank as the J arm's rank-safe projector there;
- same removed-energy fraction `‖P_S h‖²/‖h‖²` (the match is algebraic:
  one basis vector carries the required h-alignment, the rest are
  random directions orthogonal to h);
- orthogonal to the protected dictionary rows, so the control honours the
  same output-protection contract as the J arm;
- deterministically seeded per (seed_base, layer, forward, position),
  reproducible bit-for-bit.

*Why this is the complete geometric match:* the intervention is a span
projection, and a subspace's entire geometric relation to a single
vector h is characterised by its rank and the energy it removes (there
is exactly one principal angle between a subspace and a line). Matching
both therefore equates everything about the dose; the two arms differ
only in **direction content**, which is precisely the specificity claim
HP3 makes. `dynJ_rotated` was rejected as primary because rotated rows
misalign with h — selection scores and removed energy collapse toward
the isotropic regime, re-introducing the dose confound;
`dyn_spectrum_matched_nonJ` was rejected because a span projection is
invariant to dictionary row spectrum given the span. Both remain named
secondary arms available for the robustness grid.

The control consumes the J arm's logged per-position (rank, energy)
profile on the same item, so the J arm runs first and the control second
within each item block; condition order between them and the other arms
is randomised at the item level as §7 requires (the profile dependency
is on the J arm's *log*, not its outcome, and the J arm is deterministic).

**Remaining open sub-condition of the freeze:** the mechanical
dev-validation gates MC1–MC4 (rank match 100%, energy relative error
median ≤0.5% / max ≤5%, clamp rate ≤1%, protected-row cosine ≤1e-3 —
committed in `experiments/mc_dev_validation.py` before any run;
deliberately no behavioural gate, so the control cannot be tuned on
outcomes) must PASS on dev items with a primary checkpoint before any
confirmatory cell. A null against an unmatched control is not evidence
of specificity.

---

# 6 · Analysis, fixed in advance

**6.1 Primary family (Holm, α=0.05 across exactly these):** the HP1
Think-vs-Instruct interaction contrast; the HP3 Qwen tail-rate test. Two
tests, enumerated. Everything else in §3 is estimation and is not part of
the Holm family. HP5 is family B. Layer sweeps, threshold curves and
ladder checkpoints are descriptive.

**6.2 Models.**
`delta ~ model * task + baseline_accessibility + answer_token_count + (1|canonical_family)`,
with an item random intercept where items are shared across models.
Optimizer and fallback declared before outcomes; on convergence failure,
fall back to the family-clustered paired bootstrap.

**6.3 Weighting estimand — the choice, made explicitly.**
**Family-weighted (mean of family means) is the PRIMARY**, because the
scientific population is "a random relation family, then an item within
it", and because item-weighting lets `country_capital`'s 11 items
dominate a 25-family set. Item-weighted is reported as sensitivity. Both
are already computed side by side in `n2-corrected-family-pilot-v1`.

**6.3b Alias aggregation (decided pre-freeze, VM9).** The primary
answer-sequence logprob is **logsumexp over the frozen accepted-alias
set** — total probability assigned to the answer concept, which cannot
let each arm pick a different winning surface form (review §3.3 option
2). Canonical-answer lp and max-over-aliases lp are prespecified
sensitivities; per-alias rows are stored in the confirmatory parquets so
any aggregation is recomputable without a rerun.

**6.4 Tail inference.** Paired within item:
`hit_J = 1[delta_J < -1.0]`, `hit_C = 1[delta_control < -1.0]`,
statistic `mean(hit_J - hit_C)`, family-clustered bootstrap (4000 draws,
seed 4242) or a mixed-effects logistic preserving the pairing.
Specificity is never inferred from separate per-arm intervals.

**6.5 Threshold (D4).** **−1.0 nat, frozen.** Prespecified sensitivity at
−0.5 / −1.5 / −2.0. No threshold shopping; the arbitrariness is disclosed
with the curves.

**6.6 Hurdle secondary.** `Pr(tail) ~ model*task` and
`magnitude | tail ~ model*task`, separating how often the effect appears
from how severe it is — the honest description of a zero-inflated,
heavy-tailed distribution.

**6.7 Equivalence.** Real TOST returning both one-sided p-values, the 90%
interval, the SESOI and the decision.

**6.8 Descriptive secondary, prespecified.** The protected-tail
**mechanism profile** (bridge-deflation rate, protection-failure rate,
clean-rank distribution of tail vs non-tail) on the confirmatory
partition. The pilot version is the campaign's most interpretable exhibit
and its replication is itself informative.

---

# 7 · Execution and stop rules

Order: 3.1-Think (own lens) → 3.1-Instruct (own lens) → Qwen → moderators.
Complete model cells, never a thin smear.

Per model, **G4 positive control (swap injection) must pass with that
model's own finalised lens before its primary cells count**. A null on a
model whose positive control fails is assay failure, not evidence.

Stop and investigate before continuing if: baseline capability differs
from the frozen gate; a protection invariant fails once; effective rank
differs materially across primary conditions; control energy/spectrum
matching falls outside tolerance; more than the preregistered fraction of
items needs exclusion; payload hashes or model revisions disagree with the
manifest; a deterministic sentinel rerun differs beyond tolerance.

Randomise condition order within item; checkpoint per item batch; emit one
immutable per-item parquet per model/task; log requested rank, effective
rank, selected ids, scores, singular values, removed energy, phase and
protection overlap; **do not aggregate during the GPU run**; run the
locked analysis from raw rows after the final cell banks.

---

# 8 · Prohibited claims

Not sayable regardless of outcome: that the paper is confirmed or refuted
outright; that post-training monotonically creates the dissociation; that
occupancy is exactly 2 under a paper-identical estimator; that Gemma
violates differentiability or cannot be modelled by any Jacobian; that the
random arm is energy- or geometry-matched; that a J-direction causal
result is faithful beyond the measured in-band bound.

---

# 9 · Conditions outstanding before this can bind

1. **Cross-model capability cohorts** — needs every confirmatory
   checkpoint's weights (§2). *(VM9: predicate resolved as
   `capable_generation`; scoring in progress via `g5_cohorts.py`.)*
2. **Primary matched control implemented and dev-validated** (§5).
   *(VM9: implemented as `dyn_energy_rank_matched_random`, CPU
   conformance green; GPU dev-validation MC1–MC4 pending.)*
3. **A 3.1-Think and a 3.1-Instruct lens must exist.** *(Correction: the
   3.1-Instruct lens has existed since the pilot — two independent
   registered fits, `a1-ownlens-regate-olmo31instruct-v1` and
   `b1-fitB-independent-lens-olmo31instruct-v1`. Only the 3.1-Think lens
   was missing; its fit is running this block.)* No pilot 3.0 cell may
   impersonate a primary.
4. **Corrected R2 on the remaining models** for HP4's cross-model form
   (Think is done; Qwen and Instruct run this block — centered R²
   primary, raw share sensitivity, per the §0.4 estimand repair).
5. **PI sign-off on this candidate.**

Only then: one dedicated freeze commit that renames this file, runs the
family-level partition, stores hashes and assignments but no outcomes, and
tags `jspace-part2-confirmatory-freeze-v1`. Nothing else belongs in that
commit.
