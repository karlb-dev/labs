# J-space Phase 3 — living report (mechanism, generalization, paper hardening)

> **Freeze clarification.** Phase 3 was validly frozen at
> `jspace-phase3-freeze-v1`; the stale candidate header inside the historical
> preregistration is a generation defect. See
> [`../preregistration/PHASE3_FREEZE_RECORD.md`](../preregistration/PHASE3_FREEZE_RECORD.md)
> for the immutable commits, hashes, counts, and gate event.

Governing docs: `../reviews/jspace_lab_nextsteps_3_1.md` + addendum (§4/§5
govern). Phase 2 is frozen (`jspace-part2-complete-v1`); its results enter
here only as immutable `phase2-confirmatory` inputs. Every claim below
carries a tier watermark; nothing exists outside
`reports/evidence_events.jsonl`.

---

## VM10 (2026-07-29) — Phase 3 opened

### §4.1a — Protected-output selection pressure vs J damage — tier: PHASE3-DEVELOPMENT

`p3-overlap-mining-v1` · figure `p3f01` · inputs: the six frozen Phase 2
N6 parquets (3 models × {confirmatory, replication}), read-only.

The v2 ablator logs, for every position, how many PROTECTED (clean top-k)
token rows *would have entered* the J top-k and were excluded. Per item,
`blocked_rate` = blocked total / positions — selection pressure toward
the protected output region. This is the cheapest available leakage
proxy; it is NOT geometric span overlap (that is §4.1b, GPU).

**Result 1 — pressure does not explain damage, in any model.** Spearman
ρ(blocked_rate, Δlp_J): Think +0.11 [−0.04, +0.25], Instruct +0.00
[−0.13, +0.13], Qwen +0.11 [−0.04, +0.25] (family-clustered bootstrap
CIs). Tail vs non-tail blocked rates indistinguishable (MW p = 0.11 /
0.89 / 0.26). Where nonzero, the sign is *opposite* to the leakage story
(more pressure ↔ slightly less damage). Same picture against the
J-minus-matched-control specific effect.

**Result 2 — but the pressure itself is a 12× model property.** Median
blocked rate per position: Think **1.33**, Instruct **1.22**, Qwen
**0.10**. Under the identical protection contract, the OLMo lenses
constantly try to select clean-top-10 token rows (>1 per position) while
Qwen's essentially never do. OLMo's J-space is heavily output-adjacent;
Qwen's is not — a new, sharp form of the Phase 2 two-cluster picture,
and a direct motivation for the span-safe arm *specifically on OLMo*:
with that much pressure, the next-best rows selected after masking may
still overlap the protected span geometrically.

**Result 3 — the accessibility signs flip between clusters.**
ρ(clean_first_rank, Δlp_J): Think **−0.30** [−0.41, −0.17], Instruct
**−0.31** [−0.40, −0.21] — less output-locked answers lose more, the
Phase 2 accessibility organization, now CI-clean on both primaries.
Qwen: **+0.24** [+0.05, +0.40] — the opposite; damage concentrates on
items whose answers were already confidently ranked. Consistent with
composition-dependence rather than marginal accessibility.

**Threat-model calibration (what §4.1 was for):** the *within-model*
leakage account gets no prior support (pressure uncorrelated with
damage), so P3-2's decision weight shifts to the *geometric* question —
does span-safe protection change the OLMo arms, where pressure is large?
`meanJ_span_safe` remains a Phase 3 primary-family member (P3-P2);
§4.1b (principal angles, answer-direction survival) runs first thing in
the Think GPU window with `record_ids=True`.

Caveat carried: blocked_rate is a selection-pressure proxy; a uniform
geometric leak shared by every item would be invisible to Result 1 and
is exactly what §4.1b + the span-safe arm measure.

---

### N8 Level 1 — the Phase 2 primary table reproduces clean-room — tier: METHODS

`p3-n8-level1-repro-v1`. A narrative-blind agent (separate session, own
git worktree, `PYTHONPATH`-isolated, forbidden from reading any report,
registry, or figure) was handed only
`jspace_part2/protocol/N8_REPRO_PROTOCOL.md` — commands, schemas,
tolerances, no expected values — and regenerated both locked analyses
from the frozen raw parquets into an empty output root. The comparison
against published values was made afterwards, campaign-side.

| quantity | published | reproduced | Δ |
|---|---|---|---|
| confirmatory HP1 contrast | −0.504 [−0.720, −0.295] | −0.5045 [−0.7195, −0.2949] | 5e-4 |
| confirmatory HP3 Qwen CI | [0.205, 0.361] | [0.2048, 0.3608] | 2e-4 |
| replication HP1 contrast | +0.1036 [−1.681, +1.889] | +0.1036 [−1.6813, +1.8892] | 0 |
| replication HP3 Qwen CI | [0.2071, 0.3824] | [0.2071, 0.3824] | 0 |

All four inside the 2e-3 tolerance; the run wrote nothing to the
original output tree and nothing to the registry (`--no-register`
behaved as documented). The Phase 2 analysis layer is now independently
reproducible. **Level 2** (per-model sentinel item subsets) and
**Level 3** (one full GPU cell, Qwen preferred) remain open and are
release gates. The reproducer stopped before sealing its own prose
report; the JSON envelopes are the binding artifacts and are banked at
`drive://phase3/n8_level1`.

---

### §4.1b — LABEL PROTECTION IS NOT SPAN PROTECTION — tier: PHASE3-DEVELOPMENT

`p3-span-audit-olmo31-think-v2` (v1 superseded: it lacked the sharp
leakage control) · figure `p3f02` · statistics
`p3-span-audit-figure-olmo31-think-v2`.

60 frozen Phase 2 confirmatory items (18 canonical families, stratified
across tasks) on OLMo 3.1 Think with its own primary lens, scored in
Amendment-1 legacy units (2 of the 60 carry the known trailing-space
artifact; the Phase 3 default spec rejects it and caught it mid-run).
Eight arms per item. Estimates are family-weighted with
family-clustered 95% CIs, per the Phase 2 weighting convention.

**The leak is real and large.** Under label protection the selected J
span overlaps the protected span at `trace(P_J P_prot)` ≈ **0.93** of a
possible 10 (9.7% normalized), and **38.6% of all removed activation
energy lies inside the protected span**. The answer token's own
direction — never selectable, by protocol — loses ~**19%** of its norm
(survival 0.81 at L24, 0.82 at L32, 0.83 at L40). So the paper-faithful
arm has been deleting output geometry it declares protected. Span-safe
protection zeroes the overlap exactly (0.000 at every layer, survival
0.96–0.97), which is also this audit's own positive control.

**The effect shrinks by ~5× and survives.** Family-weighted mean paired
delta: label-protected **−1.40** [−2.24, −0.67] → span-safe **−0.25**
[−0.71, +0.23]. Tail rate at the frozen −1-nat threshold: **45% → 20%**.
The span-safe residual is still J-specific: its own exact rank+energy
matched control sits at **−0.005** [−0.10, +0.09], tail 2%, and the
span-safe specific tail is **+18pp**. *(Correction, VM10 late block: the
CIs first published here for the label, span-safe, and prot-energy arms
matched no registered artifact — they had been transcribed from a
mid-audit dev render. The intervals now quoted are the registered
`p3-span-audit-figure-olmo31-think-v2` values; point estimates, tail
rates, and every conclusion are unchanged, though the span-safe interval
straddles zero more widely than first stated. The same wrong intervals
appear in the 32b9840 commit message and the VM10 pause note, which are
immutable history.)*

**The sharp leakage control reproduces about a third of the mean, not
the whole thing.** The pre-existing `overlap_matched` arm matched the
projector trace but removed ≈0 energy from inside the protected span,
so it never tested the leakage account (it lands at **+0.01**, tail 2%).
The control added for this audit, `prot_energy_matched`, matches total
removed energy *and* the share taken from inside the protected span:
**−0.43** [−0.80, −0.08], tail **18%**. Reading: roughly a third of the
label arm's mean damage and 40% of its tail membership is explained by
"remove this much energy from output-adjacent geometry, content
irrelevant." The rest is not.

**The two arms damage different items** (per-item r = **0.20**). This is
a decomposition, not a rescaling — which is why §17.1's "carry both
components, never collapse them into one number" applies.

**Temporal coherence matters a little:** `persistent_matched` (one
transported frame per item-layer instead of independent per-position
seeds) gives **−0.11** [−0.19, −0.03] — small but CI-clean, so
independent per-position rotation is not a neutral choice. The Phase 2
control's name is accordingly *instantaneous* rank-and-energy matched.

**What this means for Phase 3.** §17.1's middle branch:

```
label-protected effect (−1.40)
  = protected-span leakage component  (~−0.4 reproducible by matched
                                       in-span dose; more by geometry)
  + span-safe internal-content component (−0.25, J-specific vs its own
                                       control at −0.005)
```

`meanJ_span_safe` is confirmed as the P3-P2 primary arm, and
`prot_energy_matched` is promoted to its preregistered comparator (it,
not `overlap_matched`, is the control that can falsify the content
account). Phase 2's HP3 conclusions stand as preregistered; the paper
must state that a substantial part of the label-protected tail was
output-geometry leakage, and that a J-specific tail survives full span
preservation on this development sample.

**Scope, stated plainly.** Development tier: 60 items, 18 families, one
model, and `prot_energy_matched` was designed *after* seeing the label
arm's overlap numbers — legitimate for arm selection, disqualifying for
a confirmatory claim. The confirmatory version runs on the thick paired
bank under the Phase 3 preregistration. Instruct and Qwen audits are
queued (Qwen is the informative contrast: §4.1a showed its selection
pressure is 12× lower, so its leakage component should be much smaller —
if the Qwen tail is largely span-safe-surviving, the Phase 2 HP3 result
transfers nearly intact).

---

### §4.1b cross-model — THE DECOMPOSITION IS MODEL-DEPENDENT; THE LEAK GEOMETRY IS NOT — tier: PHASE3-DEVELOPMENT

`p3-span-audit-olmo31-instruct-v1` · `p3-span-audit-qwen36-27b-v1` ·
figure `p3f03` + family-clustered statistics
`p3-span-audit-cross-model-v1`. Same design as the Think audit: the same
60 frozen Phase 2 confirmatory items (18 families), 8 arms, each model's
Phase 2 primary lens (Instruct own lens; Qwen the published n=1000
lens), Amendment-1 legacy units.

**Instrument note (BOS units on Qwen).** The Qwen audit initially
refused to construct: `ScoringSession` demanded a `bos_token_id` and
Qwen's tokenizer has none. That guard was stricter than the convention
it asserts — `jlens.from_hf(force_bos=True)` is a no-op for a tokenizer
without a BOS token, so every frozen Phase 2 Qwen artifact is in native
units. The session now scores native exactly when no BOS token exists
and records `bos_prefixed` (commit `70d8eae`, conformance test added).
**Units verified for all three audits**: every measured item baseline
reproduces a frozen N6 per-alias baseline **bit-exactly (60/60 per
model)**. Where an audit baseline differs from the N6 `lp_canonical`
column, the audit's single-alias choice (`accepted_answers[0]`) named a
different case variant than Phase 2's canonical alias — same machinery,
same units; the audit's paired deltas are unaffected (every arm scores
the same alias within item).

Family-weighted mean paired deltas (family-clustered 95% CIs) and tail
rates at the frozen −1-nat threshold:

| arm | Think | Instruct | Qwen |
|---|---|---|---|
| label-protected | −1.40 [−2.24, −0.67] · 45% | −1.81 [−2.81, −0.97] · 47% | −0.27 [−1.01, +0.42] · 30% |
| span-safe | −0.25 [−0.71, +0.23] · 20% | **−0.53 [−0.93, −0.15]** · 32% | −0.02 [−0.41, +0.40] · 15% |
| prot-energy-matched | **−0.43 [−0.80, −0.08]** · 18% | −0.15 [−0.45, +0.16] · 15% | −0.02 [−0.25, +0.20] · 7% |
| span-safe own control | −0.005 [−0.10, +0.09] · 2% | +0.01 [−0.10, +0.12] · 3% | −0.08 [−0.22, +0.06] · 2% |
| specific span-safe tail | +18pp | +28pp | +13pp |
| r(label, span-safe) per item | 0.20 | 0.29 | **0.75** |

**1 · The leak geometry is universal.** Under label protection, on every
model, the selected J span overlaps the protected span (projector
overlap, band mean: Think 0.89, Instruct 0.94, Qwen **1.37**), a large
share of removed energy sits inside the protected span (37% / 42% /
28%), and the never-selectable answer direction loses 18% / 21% / 26%
of its norm. Span-safe zeroes all of it on all three (its own positive
control). Label protection fails as span protection *everywhere*.

**2 · The behavioral leakage component is essentially a Think
phenomenon.** The exact leakage-dose control (`prot_energy_matched` —
same removed energy, same in-span share, content-free) reproduces
−0.43 of Think's label mean (~1/3), only −0.15 on Instruct (~8%, CI
straddles zero), and −0.02 on Qwen (nil). In-span dose alone does not
determine damage: Instruct has the *largest* in-span energy share yet a
small dose effect. The §4.1a selection-pressure ordering (Think 1.33 ≈
Instruct 1.22 ≫ Qwen 0.10 blocked rows/position) tracks the behavioral
dose effect, not the geometric overlap — Qwen has the *largest*
projector overlap and the *smallest* dose effect, so selection pressure
and span geometry are dissociable properties.

**3 · The content residual is largest on Instruct — and CI-clean.**
Span-safe: Instruct **−0.53 [−0.93, −0.15]**, tail 32%, specific tail
+28pp — the strongest span-safe residual of the three, consistent with
Phase 2's Instruct picture (equal-depth content-channel damage).
Think −0.25 (straddles zero as a mean; +18pp specific tail). Qwen −0.02
as a mean, tail 15% with +13pp specific tail.

**4 · A J-specific heavy tail survives full span protection on all
three models.** Every own-control tail is ≤3%; every span-safe specific
tail is ≥ +13pp. The Phase 2 tail phenomenon is not a protection
artifact anywhere.

**5 · Span-safe is a *reallocation* on OLMo and a *rescaling* on Qwen.**
Per-item r(label, span-safe): 0.20 / 0.29 on the OLMo pair — the two
arms damage substantially different items, which is why the
decomposition must be carried as two components. On Qwen r = **0.75**:
the same items lose, just less. Combined with the nil dose effect, this
says Qwen's label-protected effect was mostly content all along —
**Phase 2's HP3 transfers to span-safe protection nearly intact**, as
the §4.1a contrast predicted, while the OLMo-side label arms carried a
model-specific leakage component that the Phase 3 preregistration
removes by construction.

**Consequences for Phase 3 (unchanged, now three-model-supported).**
`meanJ_span_safe` is the P3-P2 primary arm and `prot_energy_matched`
its preregistered comparator; the paper carries the leakage/content
decomposition per model and never collapses it into one number
(§17.1 middle branch). Same development-tier scope caveats as the Think
audit; the confirmatory versions run on the thick paired bank after the
Phase 3 freeze.

---

## VM10 late block (2026-07-30) — Workstream C, banks, freeze prep

### Workstream C — the exact control on prose — tier: PHASE3-DEVELOPMENT

`p3-prose-grid-olmo31-think-v1` · `p3-prose-grid-olmo31-instruct-v1` ·
`p3-prose-grid-qwen36-27b-v1` (its first attempt OOM'd on the
151k-vocab × 1024-position logits beside 29 GB of J dictionaries; the
metrics are position-chunked — identical estimand) · figure `p3f04` +
cross-model statistics `p3-prose-grid-figure-v1`. Guard battery
v2 (`p3-guard-battery-v2`): 104 items / 8 source-pinned domains;
wikitext TEST split held out from every lens-fit corpus. Six arms per
item, including the profile-consuming exact instant rank+energy control
Phase 2 never ran on prose (§2.5), plus grammar minimal pairs and a
pre-registered greedy generation audit.

Mean Δ NLL per token (all 84 text items):

| arm | Think | Instruct | Qwen |
|---|---|---|---|
| label-protected J | +0.175 | +0.530 | **+0.945** |
| span-safe J | +0.089 | +0.150 | +0.209 |
| exact instant matched | **+0.003** | **+0.002** | **+0.021** |
| prot-energy matched | +0.076 | +0.123 | +0.052 |
| mechanics random | +0.078 | +0.120 | +0.104 |
| logit-protected | −0.021 | −0.054 | +0.045 |

**Reading (R4/§7.5).** Three findings:

1. **Prose cost is J-content everywhere, never dose**: the exact
   matched control sits at +0.002 to +0.021/token on all three models
   while the label arm runs +0.175 to +0.945.
2. **The prose-cost ordering INVERTS the task ordering and tracks the
   §4.1b geometry.** Qwen — smallest label-arm task effect (−0.27
   family-weighted) — has the LARGEST prose cost (+0.945/token), and
   the cross-model prose ladder (Qwen > Instruct > Think) follows the
   projector-overlap ladder (1.37 > 0.94 > 0.89), not the
   selection-pressure ladder. On prose every position leaks; span-safe
   removes 72–78% of the label cost on all three (Qwen +0.945 → +0.209).
3. **No model earns "selective" wording at this tier.** The §7.4 index
   (standardized task effect − standardized prose effect, label arm,
   both components carried): Think +1.57, Instruct +1.50, Qwen +1.25 —
   standardized prose damage exceeds standardized task damage on ALL
   three. Per §7.5 the sentence remains "nonspecific J-channel
   vulnerability cannot be excluded" for every primary, with the
   sharpened qualifier that the vulnerability is content-specific
   (exact-control-clean) and predominantly output-span leakage. The
   confirmatory-era selectivity computation reruns on the span-safe
   arm set after the freeze.

Grammar preference survives every arm (85–100%; baselines 95–100%);
damage is diffuse across all 8 domains, not domain-selective.

### G5, the intersection squeeze, and two instrument catches — tier: PHASE3-DEVELOPMENT

Banks: **F v7** (204 bundles / 48 families, 204/204 verified against
the per-revid reference; v7 appended adjectival currency aliases,
alias[0] stable) and **S v3** (120 bundles / 24 template families) —
the 72-family target authored. G5 scored all 972 items on each primary
(`p3-g5-bank-*`).

Two instrument catches, both caught by reading failed generations:

1. **`ScoringSpec.normalize` deleted whitespace runs** instead of
   spacing them, so `" the\nBaht"` normalized to `"thebaht…"` and
   failed prefix grading on *correct* generations (commit with
   regression test; offline regrade from stored generations — the
   parquets carry each item's greedy text).
2. **The capability predicate itself**: models often answer in
   exam-continuation mode ("… which country?\n----\nAnswer: Hungary"),
   correct but not a prefix. Since every endpoint is a teacher-forced
   logprob, the inclusion predicate was set (pre-freeze, no intervention
   outcomes involved) to **answer contained in the 8-token greedy
   continuation**, with the stricter prefix rate kept as a reported
   column (`p3-g5-bank-*-v{4,3,3}`).

Cohorts under the final predicate: capable rates 73.5/74.2/77.9%
(Think/Instruct/Qwen); both-variant-capable facts 161/156/189;
**three-model intersection 104 facts / 34 families** (18 F + 16 S).
**Disclosed R2 deviation:** the ≥25/side intersection floor is
infeasible with this bank generation — the Phase 2 burn list (858
consumed answers) forces less-famous instances exactly where
three-model capability is scarcest. The frozen floor is measured
feasibility (≥16/side), and P3-P1's power at that size is disclosed in
the preregistration. Split preview (`p3-family-split-preview-v1`, seed
4242): 36/36 families, F 24/24, intersection 16/18, all standardized
imbalances ≤0.35, disjointness clean.

### P3-P3 model resolution and power design — tier: PHASE3-DEVELOPMENT

Bridge identifiability gates (magnitude-blind: coverage +
measurability only): Think `identifiable=true` (any-piece coverage
0.85, 95% of rescue contrasts above the determinism floor), Qwen
`identifiable=true` (0.85, 95%). The tie-break rule — pinned in the
prereg *before* the Qwen gate ran — sends exact coverage ties to Qwen:
**P3-P3 = Qwen**, bridge arms enabled in its primary-grid config.

Power (`p3-power-sim-v3`; v1 and v2 were superseded after their own
outputs exposed generator defects — v1 ignored within-fact
cancellation entirely, v2's P3-P2 "null" leaked a +0.17 tail-rate
difference): null-calibrated (rejections 1.3–1.5% at α/3 = 1.67%);
**P3-P2 MDE ≈ 0.2 tail-points at 30–36 families** (the Phase 2 anchor
effect was +0.279); **P3-P1 is powered only for ≥0.3–0.5-nat effects**
depending on realized within-fact cancellation — carried as a test
with that disclosure, with the named estimation targets bearing the
quantitative story regardless (the Phase 2 binary-primary lesson).

The preregistration candidate has **zero unresolved PENDING markers**;
the freeze runs when the Qwen Workstream C cell banks (its registry
event is a freeze gate).

---

## Block B — THE PHASE 3 PRIMARY RESULTS — tier: PHASE3-CONFIRMATORY

Cells `p3-primary-grid-{olmo31-think,olmo31-instruct,qwen36-27b}-v1`
(170/164/188 frozen-cohort items, §7 stop rule passed on 25 baselines
per model against the G5 manifest, randomized arm order, no
aggregation viewed) · locked analysis `p3-locked-analysis-v1` (run
once, from raw parquets, after the last cell) · figure `p3f05`.

**The Holm family:**

| primary | estimate | p (Holm) | verdict |
|---|---|---|---|
| **P3-P2** span-safe J-specific tail, Qwen | +0.096 tail-rate excess over the exact matched control (188 items / 26 families) | ~1e-5 (< α) | **REJECTS** |
| **P3-P3** bridge-protection rescue, Qwen | +0.431 nats (true vs distractor bridge protection, 94 composed items) | 0.018 (< α) | **REJECTS** |
| **P3-P1** thick task contrast, Qwen − OLMo pair | −0.261 nats (wild-cluster 95% CI [−0.52, −0.01]; se 0.130; 17 intersection families) | 0.062 (> α) | directional, under the disclosed MDE |

**What this establishes.**

1. **The J-specific heavy tail survives full span-safe protection at
   confirmatory tier.** Phase 2's HP3 was measured under label
   protection, which the §4.1b audits showed leaks 28–42% of removed
   energy into the protected span; P3-P2 re-establishes the phenomenon
   with the leak *removed by construction* — the campaign's central
   specificity claim now rests on the clean arm. The surviving effect
   (+0.096) is about a third of the label-protected size (+0.279),
   quantitatively consistent with the development decomposition.
2. **Bridge mediation is confirmed on Qwen.** Protecting the true
   bridge's token directions rescues composed-item damage by +0.43
   nats relative to protecting a matched counterfactual bridge — the
   §18.5 "awaiting Phase 3" item "bridge deletion mediates the tail"
   moves to established-under-the-assay, on the model whose gate was
   chosen magnitude-blind.
3. **P3-P1 lands exactly where the disclosed power analysis said it
   could:** paper-shaped direction (Qwen's composition penalty exceeds
   the OLMo pair's by 0.26 nats; item- and relation-group-weighted
   sensitivities agree: −0.18 / −0.22), p=0.062 at 17 intersection
   families — beneath the preregistered α, beneath the scenario MDEs.
   Per the prereg's own §5 statement, the estimation targets carry the
   quantitative story: this is an honest near-miss, not a null.

**The estimation targets:**

- **Think vs Instruct on the thick bank: indistinguishable** (+0.01
  [−0.45, +0.39], 22 families) — the thick-bank version of Phase 2's
  knife-edge HP1 quantity shows the 3.1 pair as one cluster,
  consistent with their flat capacity.
- **The §4.3 workspace-word test: Qwen's contrast does NOT appear on
  Bank S** (+0.02 [−0.33, +0.42]) while its Bank F behavior drives
  P3-P1's direction. Per the addendum's pre-stated decision rule, the
  honest noun for what Qwen's J-channel carries is **"knowledge-access
  channel"** — the working-memory reading of "workspace" is not
  supported at this tier. (Think shows a CI-clean *positive* Bank-S
  composition term, +0.15 [+0.04, +0.26] — composed in-context items
  lose *less* than direct ones under span-safe J — flagged for the
  mechanism block, not interpreted further here.)

**Reproduction ladder:** N8 Level 2 sentinels PASS on all three models
— 20 items × all conditions re-measured under the frozen Phase 2
commands, worst deviation **0.0 (bit-exact)** on every lp column
(`p3-n8-level2-*-v1`). Level 3 (one full clean-room Qwen cell) runs in
the current window.

### Replication side + N8 completion — tier: PHASE3-REPLICATION / METHODS

`p3-replication-grid-*-v1` (reduced arm set on the untouched 36-family
replication partition) · `p3-replication-analysis-v1`:

- **P3-P2 REPLICATES**: span-safe J-specific tail excess **+0.102**
  (p ≈ 1e-5, 28 families) vs confirmatory +0.096 — the campaign's
  central specificity claim is confirmed AND replicated on disjoint
  held-out families under the leak-proof arm.
- P3-P1 replication: −0.197, p = 0.218 (17 intersection families) —
  same direction at ~3/4 the confirmatory size, same disclosed power
  limitation; the two sides agree in sign and rough magnitude.

**N8 Level 3 PASSES** (`p3-n8-level3-qwen36-27b-v1`): the FULL 184-item
Qwen confirmatory cell re-measured clean-room under the frozen Phase 2
command — worst lp deviation **0.0 (bit-exact)**. With Level 1
(analysis layer, ≤5e-4), Level 2 (three bit-exact sentinels), and
Level 3 (one full cell), the reproduction ladder specified as a
public-release gate is fully discharged.

---

## §6.5 mediation factorial — QWEN CONSUMES ITS BRIDGE CHANNEL; THINK DOES NOT — tier: PHASE3-DEVELOPMENT

`p3-bridge-mediation-qwen36-27b-v1` (40 composed confirmatory items) ·
`p3-bridge-mediation-olmo31-think-v1` (20-item spot check). Seven arms
per item, span-safe base protection; bridge piece sets with tokenizer
variants; family-clustered CIs.

| arm (Δ vs baseline, nats) | Qwen | Think |
|---|---|---|
| span-safe reference | −0.19 [−0.66, +0.22] | −0.57 [−1.17, −0.07] |
| + true-bridge protected | **+0.12** [−0.18, +0.40] | −0.51 [−1.13, +0.00] |
| + distractor-bridge protected | −0.56 [−1.06, −0.10] | −0.30 [−0.94, +0.20] |
| bridge-only lesion (output preserved) | **−0.89** [−1.58, −0.30] | −0.19 [−0.32, −0.07] |
| counterfactual-bridge swap | **−4.05** [−6.33, −1.92] | −0.09 [−0.40, +0.36] |
| answer-only lesion (diagnostic) | −2.53 [−3.16, −1.91] | −4.48 [−5.24, −3.62] |
| unrelated-content lesion | −0.04 [−0.12, +0.05] | −0.14 [−0.22, −0.06] |
| **rescue contrast (true − distractor)** | **+0.67 [+0.29, +1.11]** | **−0.20 [−0.38, −0.04]** |

**The double dissociation, in §6.8's own vocabulary:**

- **Qwen: strong bridge mediation.** Deleting only the bridge's
  directions — output fully preserved — damages composed answers
  (−0.89, CI-clean). Substituting the counterfactual bridge's direction
  at matched energy is *catastrophically worse than deletion* (−4.05):
  the model demonstrably READS what sits in that channel and follows
  it. The unrelated-content lesion is nil (−0.04), so this is
  content-specific; true-bridge protection removes the composed damage
  entirely. The preregistered P3-P3 rescue (+0.43 confirmatory)
  reproduces here at +0.67.
- **Think: no bridge mediation — mildly inverted.** The swap does
  nothing (−0.09 ≈ 0): injecting a wrong bridge into the channel is
  inert, so whatever Think's composed-item computation uses, it is not
  the readable bridge content. Its rescue contrast is slightly
  *negative* (−0.20 [−0.38, −0.04]), and its unrelated-content lesion
  is nonzero — less content-specific damage overall, consistent with
  the accessibility-organized cluster.

This is the mechanism-level counterpart of every cross-model split in
the campaign: the same intervention grammar that shows Qwen routing
composed answers *through* readable bridge content shows Think not
doing so. Scope: development tier (arm set beyond the preregistered
P3-P3; Think n=20), one composed-item slice; the answer-only
diagnostic behaves on both models.
