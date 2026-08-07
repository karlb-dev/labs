# A5 sub-analysis — the H6 ↔ Phase-3 estimand reconciliation

**Registered:** `analysis-h6-phase3-reconciliation-v1`
(`reports/analysis_events.jsonl`), predeclared by
`paper_analysis_addendum.md` §2.2 and written before any paper drafting.
This document states two frozen estimands formally and derives which
conclusions each bounds. It computes no new number; every quantity cited
is a frozen registered value.

## The objection this answers

> "Your own H6 says no in-band finite-dose linear regime exists on OLMo
> (L24/L32/L40, ε ∈ [0.001, 0.10]), so how do your in-band Phase 2/3
> interventions mean anything?"

The answer is not rhetorical; the two experiments estimate different
functionals of the model, with different success criteria, and the
validity evidence for the intervention estimand is in-situ.

## Estimand 1 — H6: the lens as a finite-dose linear predictor

Fix checkpoint $M$, prompt $p$, layer $\ell$, position, direction family
$d$, and relative dose $\varepsilon$. Let $F$ denote the frozen
source-to-target residual map and $J_{p,\ell}$ its exact prompt-specific
Jacobian (dual exact-JVP backends under the imported calibrated envelope,
ceiling $0.07870368901355948$). H6 asks whether the **first-order
prediction**

$$F(h + \varepsilon u) - F(h) \approx \varepsilon\, J_{p,\ell}\, u$$

meets a frozen quantitative gate (tangent cosine / relative error, with
backend, delivery-fidelity, and SNR eligibility conditions and a 0.90
cell passage floor), on a strict prospective ladder
($\varepsilon \in \{0.001, \dots, 0.10\}$, layers 24/32/40/56, three
direction families, four prompts, 336 rows per checkpoint).

**Frozen verdict** (`ol2-transport-validation-joint-v1`): no L24/L32/L40
layer–dose cell reaches the floor at either checkpoint; OLMo-3.1 Think
passes only L56 at $\varepsilon = 0.10$ (12/12); Base does not (9/12).
Route: `h6_fail_in_band_with_checkpoint_specific_late_anchor`.

**What this bounds.** Any design that *consumes the lens as a linear
predictor of finite responses at specified doses*: crossed-lens patching,
dose-transport extrapolation, "inject $\varepsilon u$ and read the
predicted downstream state" designs, and any claim of a licensed
finite-dose linear-response regime in the assay band. On the tested
ladder, none of these is licensed on OLMo in-band.

## Estimand 2 — Phase 2/3: span removal against an exact dose-matched control

Fix item $i$ with per-position hidden states $h_{t}$. The intervention
removes the span of a selected, output-protected J-derived dictionary
subset $S_i$ (span-safe in Phase 3: selection and geometry are excluded
from the protected output span by construction), realizing at each site a
projection of effective rank $r_{i,t}$ removing energy fraction
$e_{i,t} = \lVert P_{S}h_t\rVert^2 / \lVert h_t\rVert^2$. The comparator
removes a **random subspace matched exactly, by construction, to
$(r_{i,t}, e_{i,t})$ at the same site**, orthogonal to the protected
rows. The estimand is the paired behavioral contrast

$$\Delta_i = \mathrm{score}_i(\text{J-span removal}) -
\mathrm{score}_i(\text{matched random removal}),$$

aggregated by frozen family-clustered statistics (P-HP3 tail-rate
difference; P3-P2 tail excess at $-1$ nat).

**Frozen verdicts:** P-HP3 $+0.2788$ $[0.2048, 0.3608]$ (Holm
$p = 5\times10^{-4}$), replication $+0.2966$; P3-P2 $+0.095833$
(plus-one $p = 1/100001$), replication $+0.102083$
(`n6-confirmatory-analysis-v2`, `n6-replication-analysis-v2`,
`p3-inference-audit-v1`).

**Where its validity evidence lives.** Not in a linearity premise — in
the behavior of the controls at the same sites and doses:

1. the exact rank-and-energy-matched control produces approximately no
   damage and no $-1$-nat tail (registered control tails ≈ 0; e.g. the
   registered cross-model span audit shows matched-control tail rates
   0.000–0.017 against J-arm tail rates 0.15–0.32 on the same items);
2. G4 positive controls pass per model (flip rates 0.76/0.76/0.78 vs
   0.18–0.24 for random swaps), so the instrument can move behavior when
   it should;
3. the effect replicates on frozen held-out families.

## Why H6's failure does not transfer

1. **Different functional.** H6 evaluates the *accuracy of a linear
   approximation to $F$* along injected directions at specified
   $\varepsilon$. The ablation estimand never invokes that approximation:
   removal is applied to the actual forward pass, and its consequence is
   *measured*, not predicted. No step of the Phase 2/3 inference chain
   passes through $J\,u$ as a predictor.
2. **Dose semantics differ.** H6 doses are relative-$\varepsilon$
   injections along frozen direction families. Ablation doses are the
   achieved per-site $(r, e)$ removal profiles. The frozen mapping
   between the two was never collectable: the registered archives lack
   exact site-level total-dose records (`ol2-transport-validation-joint-v1`
   §dose audit), so in-band/out-of-band placement of the causal doses is
   **unavailable, not adverse** — and asserting either placement is
   forbidden.
3. **Nonlinearity is dose-symmetric in the contrast.** Whatever
   higher-order structure $F$ has at these sites acts on both arms of
   $\Delta_i$ at exactly matched $(r, e)$. A generic finite-scale
   nonlinearity inflates or deflates both arms together; what survives
   the subtraction is the *direction-content difference* between the
   J span and a random span — which is the claim, not a confound. (A
   nonlinearity that couples specifically to J-span content would be a
   *mechanism* of the measured effect, not an invalidation of it.)
4. **The lens's role in the ablation chain is selectional.** The lens
   proposes which directions to remove; the causal claim is about the
   consequence of removing them. A lens can be an unlicensed finite-dose
   predictor and still select a causally load-bearing span — exactly the
   asymmetry Q-L4 froze on Qwen (aggregate causal endpoints fit-stable
   while sparse selections are not; `p4-qwen-canonical-lens-decision-a1000-dev-v1`).

## What each paper may and may not say

- Paper A (empirical) MAY: report the paired ablation effects with their
  in-situ control evidence; cite this reconciliation in limitations.
  MUST NOT: claim any licensed finite-dose linear-response regime
  in-band on OLMo, or map causal doses onto the H6 ladder.
- Paper B (methods) MAY: use H6 as the checkpoint/dose applicability
  boundary for lens-as-predictor designs; present this reconciliation as
  the estimand-boundary section. MUST NOT: describe H6 as invalidating
  (or validating) the paired ablation effects.
- Neither paper may treat the missing site-dose archive as either
  coverage or non-coverage (Phase 5D is the prospective fix).

## One-sentence form for both papers

> H6 bounds the lens **as a finite-dose linear predictor** on a
> prospective ladder; the causal result is a **measured removal
> contrast against an exactly dose-matched control**, whose license is
> the control's behavior, not a linearity premise — the two estimands
> share a model and a band, not an inference chain.
