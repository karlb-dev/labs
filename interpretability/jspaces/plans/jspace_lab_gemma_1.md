# jspace_lab_gemma_1.md

## Gemma transport side study 1: exact JVP validation, curvature localization, and nonlinear recovery

**Purpose:** chase the Gemma 4 31B transport anomaly to a defensible mechanism-level conclusion. The goal is not to force Gemma into the OLMo/Qwen model matrix. The goal is to determine exactly why a fixed mid-band Jacobian lens fails, whether the failure is numerical, finite-radius curvature, context-dependent tangent cancellation, a late basis conversion, or a particular architectural sublayer, and whether any scoped alternative transport instrument is valid.

**Starting boundary:** fork from the clean Phase 4 branch boundary `3b041735d8b842de46a9c0a474fccd0c44e0841a`. Import all prior Gemma pilot artifacts and OLMo positive-control artifacts by hash. Treat the supplied `gemma4_nonlinear_jacobian_handout` as the governing mathematical and claim-boundary document. This side track is methods/development work. It must not edit the Phase 4 package or registry, and it must not open Phase 4 untouched outcome families.

**Recommended branch and namespace:**

```text
branch:   interp_jspace_gemma_transport
package:  interpretability/jspace_gemma/
run root: /content/drive/MyDrive/interpret/special-lab-1/gemma_transport_<date>/
registry: interpretability/jspace_gemma/reports/evidence_events.jsonl
prefix:   gm-
```

> **Paste-line for the coding/research agent**
>
> Read the full Gemma nonlinear-Jacobian handout, its source TeX, `REPORT_PART2.md`, the prior Gemma fit/readout/identification/local-linearity/faithfulness code and results, the OLMo positive-control results, and the current Phase 4 conclusion skeleton before writing code. This is a bounded transport autopsy, not a search for a positive workspace result. Create an isolated `jspace_gemma` package, run root, and registry from the exact clean parent commit. Import prior evidence read-only and explicitly supersede no old artifact in place. First build and validate an exact directional-autodiff harness: exact autograd JVP versus faithfully delivered fp32 secants over a frozen epsilon ladder, with the identical code run on an OLMo positive-control checkpoint. Do not call a finite difference an exact JVP, and do not interpret Gemma until the OLMo control and tiny-model goldens pass. Then localize the first-order defect layer by layer and after every attention, normalization, and MLP substage; run frozen-routing, norm-linearization, and gate-linearization interventions; separate prompt-specific tangent accuracy from corpus-average cancellation using fixed probe directions and held-out contexts; test the late L44-L52 band; and only then try path-integrated Jacobians, directional second-order corrections, or nonlinear probes. Every conclusion must distinguish readout opacity from transport nonlinearity, differentiability from finite-dose validity, and a methods boundary from absence of a workspace. Commit at each evidence boundary and finish with a state-of-record report plus `IMPORT_BUNDLE_PHASE4.json`.

---

# 0. Executive verdict and scientific target

## 0.1 What is already known

The prior Gemma campaign established several important but incomplete facts:

1. **Mid-band output opacity is real.** Through much of the relative depth band used by the original workspace assay, the accepted answer remains extremely low-ranked in the final output basis. The rank collapses only around the late L42-L44 transition and reaches near-final readability around L52.
2. **A fitted J-lens does not rescue the mid-band readout.** At identified layers in the paper-relative band, the fitted J readout can be worse than the plain logit lens.
3. **Lens identification and validity are different.** Four independent 30-prompt fits recover incompatible maps at L22, so no model-level readout claim is licensed there. From roughly L30 upward, slice maps agree strongly, but that only shows a reproducible average map, not a faithful finite-dose model.
4. **The corrected fitted-J faithfulness test remains poor.** After fixing the earlier estimand mismatch, Gemma’s fitted transport does not become faithful in the way OLMo’s does at late layers.
5. **The existing superposition test suggests nonlinearity.** Homogeneity and additivity defects worsen or remain large at intervention-relevant scales on Gemma while an OLMo control approaches linear behavior.
6. **The existing evidence is not the gold-standard closure.** The current local-linearity script uses finite responses without an exact prompt-specific autograd JVP. It cannot fully separate tangent error, higher-order curvature, mixed-precision delivery, hook-path mismatch, and average-map cancellation.

The handout therefore states the correct remaining decisive experiment:

> Compare an exact prompt-specific autograd JVP with faithfully delivered finite secants over a control-calibrated fp32 epsilon ladder.

This side study begins there.

## 0.2 What the result must not be allowed to mean

Even a clean Gemma nonlinearity result does **not** establish:

- that Gemma is non-differentiable;
- that backpropagation is invalid;
- that Gemma has no workspace or no verbalizable internal state;
- that all Gemma sizes or checkpoints share the geometry;
- that a nonlinear probe cannot decode the state;
- that the original J-lens is invalid on models where its transport premise passes;
- that the final logit softcap causes residual-to-residual curvature;
- that late readout success licenses a mid-band causal J-space intervention.

The intended conclusion is instrument-specific:

> At specified layers, prompts, directions, and finite intervention radii, a fixed context-independent linear transport map is or is not an adequate model of the downstream response.

## 0.3 The five questions this study must close

### GQ1. Is the previously measured defect real or an implementation/numerical artifact?

Exact JVP versus the smallest faithfully delivered secant answers this.

### GQ2. Is the failure within-context finite curvature or between-context tangent heterogeneity?

Prompt-specific JVP accuracy versus mean and clustered maps answers this.

### GQ3. Which architectural operations create the defect?

Layer/sublayer localization, frozen routing, normalization linearization, and MLP gate decomposition answer this.

### GQ4. Does a valid transport band exist late in the network?

The exact gate at L44-L52 answers this.

### GQ5. Can a scoped nonlinear alternative recover predictive transport without pretending that one global token dictionary still exists?

Path-integrated JVP, directional second order, context atlases, and nonlinear readout tests answer this.

## 0.4 Priority order

```text
G0  isolated package, architecture/hook audit, immutable imports
G1  exact JVP versus secant gate with OLMo positive control
G2  layerwise and sublayer defect localization
G3  attention-routing freeze and local/global block tests
G4  RMSNorm, QKNorm, and gated-MLP linearization tests
G5  prompt-specific versus mean versus clustered tangent study
G6  late-band transport and workspace-readout assay
G7  path-integrated and second-order recovery
G8  nonlinear probe as readout-only fallback
G9  state-of-record, reproduction, and import bundle
```

G1 is never optional. If G1 cannot be made valid, the side study stops with a methods blocker rather than substituting another secant plot.

---

# 1. Foundation and immutable evidence imports

## 1.1 Package tree

```text
interpretability/jspace_gemma/
├── README.md
├── pyproject.toml
├── constraints.txt
├── configs/
├── data/
├── jspace_gemma/
│   ├── __init__.py
│   ├── __main__.py
│   ├── architecture.py
│   ├── autodiff.py
│   ├── hooks.py
│   ├── imports.py
│   ├── linearizations.py
│   ├── manifests.py
│   ├── paths.py
│   ├── provenance.py
│   ├── registry.py
│   ├── repro.py
│   ├── stats.py
│   └── experiments/
├── protocol/
├── preregistration/
├── reports/
│   ├── evidence_events.jsonl
│   └── figures/
├── release/
├── reviews/
└── tests/
```

## 1.2 Foundation event

Register:

```text
gm-foundation-v1
```

Pin:

- parent commit and side branch;
- exact Gemma checkpoint revision and model snapshot inventory;
- tokenizer and processor revisions;
- exact text-decoder path inside the multimodal wrapper;
- architecture manifest;
- prior Gemma lens hashes and fit manifests;
- prior local-linearity, fitted-J faithfulness, identification, readout, and deep-band results;
- OLMo positive-control model/lens/results;
- exact CUDA/runtime environment;
- side run root and registry;
- package source inventory;
- claim boundary and forbidden claims;
- tests.

## 1.3 Prior evidence to import

At minimum:

```text
a3-gemma-fullfit-v1
a3-gemma-identification-v1
a3-gemma-readout-verdict-v1
local-linearity-v3-gemma4-31b
linearization-faithfulness-gemma4-31b-v2
readout-control-olmo3think-v1
local-linearity-v3-olmo3think
linearization-faithfulness-olmo3think-v2
```

Also import the exact source files that generated them. The side report must distinguish historical results, current live evidence, and superseded diagnostic attempts.

## 1.4 Architecture manifest

Verify from the exact model config and module graph rather than family memory:

- number of decoder layers;
- residual width and MLP width;
- local/full-attention schedule;
- sliding-window size;
- query/KV head counts and head dimensions by block type;
- pre- and post-attention RMSNorm;
- pre- and post-MLP RMSNorm;
- QKNorm location;
- gated MLP activation and branch equations;
- whether global attention reuses keys as values;
- RoPE conventions;
- tied embedding/unembedding;
- final output softcap location;
- per-layer embedding inputs disabled/enabled;
- exact hook points and residual semantics.

Emit:

```text
configs/gemma4_31b_architecture_manifest.json
```

The expected high-level pattern is five sliding-window blocks followed by one full-attention block, repeated. The code must derive and verify the actual indices.

## 1.5 Softcap correction

The final `30*tanh(z/30)` output softcap is after the language-model head. It can affect logit magnitudes and logit-space derivatives. It cannot explain:

- residual-to-residual nonlinearity measured before unembedding;
- answer-rank opacity, because a strictly increasing elementwise transform preserves rank.

Therefore:

- final residual is the primary transport target;
- pre-softcap logits are a secondary target;
- post-softcap logits are a tertiary audit with the softcap derivative included;
- no “pre-cap residual target” terminology.

---

# 2. Mathematical estimands and diagnostics

## 2.1 Downstream map

For prompt/context `x`, source layer `l`, source position `t`, and downstream target checkpoint `k`, define:

```text
F_{x,l,t→k}(z)
```

as the downstream activation after replacing the clean source residual `h` by `z` and running the exact remaining model path.

For direction `v` and scale `epsilon`:

```text
Delta_x(epsilon; v) = F(h + epsilon v) - F(h)
```

The exact prompt-specific tangent is:

```text
j_x(v) = D F_x(h) v
```

## 2.2 Tangent accuracy

For each finite epsilon:

```text
cos_tan(epsilon) = cos(Delta(epsilon), epsilon * j_x(v))

gain(epsilon) = ||epsilon * j_x(v)|| / ||Delta(epsilon)||

e_rel(epsilon) = ||Delta(epsilon) - epsilon*j_x(v)||
                 / (||Delta(epsilon)|| + eta)
```

Report all three. Cosine alone can hide magnitude failure; relative error alone can look bad when the response is tiny.

## 2.3 Vector homogeneity defect

```text
H(epsilon; v) = ||Delta(2epsilon;v) - 2 Delta(epsilon;v)||
                / (2||Delta(epsilon;v)|| + eta)
```

Also report response cosine and scale ratio for continuity with the earlier harness.

## 2.4 Odd-symmetry defect

```text
O(epsilon; v) = ||Delta(epsilon;v) + Delta(-epsilon;v)||
                / (||Delta(epsilon;v)-Delta(-epsilon;v)|| + eta)
```

For a purely linear map this is zero. It estimates even-order directional curvature.

## 2.5 Additivity defect

For normalized directions `v,w`:

```text
A(epsilon; v,w) = ||Delta(epsilon;v+w)
                    - Delta(epsilon;v)
                    - Delta(epsilon;w)||
                  / (||Delta(epsilon;v+w)|| + eta)
```

Use orthogonalized pairs and pairs with controlled cosine to probe cross-curvature.

## 2.6 Curvature-versus-floor fit

For each prompt, layer, position, target, and direction, fit over the faithful epsilon range:

```text
e_rel(epsilon) ≈ a + b * epsilon
```

Interpretation:

- `a > 0`, `b ≈ 0`: implementation mismatch, quantization floor, or tangent bias;
- `a ≈ 0`, `b > 0`: first-order tangent is valid infinitesimally and finite curvature grows with radius;
- both large: mixed bias and curvature;
- neither resolvable: insufficient response SNR.

Use robust regression and bootstrap prompts, not one pooled least-squares line dominated by a few high-response cells.

## 2.7 Context heterogeneity

For fixed probe directions `V=[v_1,...,v_q]`, collect prompt-specific directional maps:

```text
Y_x = [J_x v_1, ..., J_x v_q]
```

Define:

- pairwise prompt-map CKA or cosine;
- mean-map prediction error on held-out prompts;
- context cancellation index;
- within-cluster versus between-cluster variance;
- cluster-map held-out prediction.

One useful cancellation index is:

```text
CCI = E_x ||Y_x - mean(Y)||_F^2 / (E_x ||Y_x||_F^2 + eta)
```

The exact formula and centering should be frozen before target-model results.

---

# 3. G1: exact autograd JVP versus finite secant

## 3.1 Why the current scripts are insufficient

The prior `local_linearity.py` is useful but not decisive. It:

- uses finite responses only;
- injects one constant direction at all valid positions;
- targets a summed final residual;
- runs the model in mixed precision;
- uses coarse homogeneity/additivity rules;
- does not calculate an exact prompt-specific JVP;
- cannot isolate a tangent mismatch from higher-order curvature.

The prior `linearization_faithfulness.py` correctly matches the position-averaged, target-summed object fitted by `jlens`, but it tests a fitted corpus-average J, not the exact tangent. It therefore answers estimator faithfulness, not the existence of a local tangent regime.

G1 supersedes neither artifact historically. It provides the missing gold-standard layer.

## 3.2 JVP backend ladder

Implement and validate in this order:

### Backend A: forward-mode `torch.func.jvp`

Functionalize the downstream suffix from an explicit source activation tensor and compute directional derivatives directly.

### Backend B: `torch.autograd.functional.jvp`

Use PyTorch’s fallback where custom operations do not support forward AD. Record whether it uses reverse-over-reverse and the memory cost.

### Backend C: local unfused replacement

If a specific fused attention/norm operation lacks JVP support, replace only that operation with a mathematically equivalent eager implementation. Before scientific use, require:

- clean forward parity;
- ordinary first-order gradient parity;
- secant parity at a moderate epsilon;
- OLMo positive-control parity where applicable.

### Forbidden fallback

Do not label a central finite difference as “exact JVP.” A high-accuracy secant may be a sensitivity, but failure to obtain autodiff is a methods blocker that must be reported.

## 3.3 Functional suffix design

Hooks that mutate module outputs are convenient for secants but awkward for functional autodiff. Prefer an explicit downstream function:

```python
def downstream_from_source(
    h_source: torch.Tensor,
    *,
    cached_prefix_state: PrefixState,
    source_layer: int,
    target_spec: TargetSpec,
) -> torch.Tensor:
    """Run layers source_layer+1 ... target from explicit h_source."""
```

Requirements:

- exact attention mask and position IDs;
- exact cache/no-cache convention;
- no detached tensors on the derivative path;
- source state is the only differentiable primal unless testing component derivatives;
- target extraction has a deterministic shape;
- clean functional suffix matches the full-model clean target within a frozen tolerance;
- supports one source position and a uniform valid-position perturbation as separate modes.

## 3.4 Mixed-precision contract

The model may remain bf16 for memory, but the perturbation contract is fp32-delivered:

1. capture clean source state in fp32;
2. construct desired perturbation in fp32;
3. apply using the exact model path;
4. record the realized perturbation after dtype conversion;
5. judge only cells with:
   - cosine(realized, desired) >= 0.999;
   - relative norm error <= 0.01;
6. use realized, not desired, direction/scale in secant comparisons where quantization changes it;
7. calculate JVP through the same cast/path used by the functional suffix.

If the smallest epsilon is not faithfully delivered, it is unmeasurable. Increase epsilon rather than calling it nonlinear.

## 3.5 Source perturbation modes

### Primary: single-position

Matches the position-wise causal ablation. Use selected source positions such as final prompt token, bridge token, subject token, and controlled interior token.

### Secondary: uniform valid-position

Matches the corpus-fitted J estimand and the earlier harness. This is expected to be more linear in attention because all keys may move together.

### Optional: local window

Perturb a 3- or 5-token region to interpolate between single-position and uniform modes.

The paper-facing transport license for position-wise intervention must use the single-position result.

## 3.6 Prompt set

Stage 1 gate:

- 4 prompts;
- one factual, one multi-hop/bridge, one neutral prose, one code/SQL-like prompt;
- lengths spanning short and medium contexts;
- no Phase 4 untouched facts.

Stage 2 development set:

- at least 16 prompts;
- 4 broad task strata;
- relation-family diversity;
- fixed before Stage 1 Gemma outcomes are interpreted beyond harness validity.

## 3.7 Layers and targets

Initial layers:

```text
L22  shallow paper-band edge, historically not identified by mean fit
L30  identified mid-band
L37  upper paper-band edge
L42  readout transition neighborhood
L44  late transition
L48  late readable band
L52  near-output readable band
```

Add matched OLMo relative-depth controls.

Targets:

1. residual after each downstream block;
2. final residual before final norm;
3. normalized final residual;
4. pre-softcap logits on a fixed token subset;
5. post-softcap logits as an audit.

Do not begin with full-vocabulary JVP outputs. Use residual targets and selected logits.

## 3.8 Direction set

Per prompt/layer/position:

- random Rademacher directions;
- random Gaussian directions;
- radial activation direction `h/||h||`;
- directions tangent to the constant-norm sphere, orthogonal to `h`;
- fitted-J selected token directions;
- accepted-answer direction;
- true bridge direction where applicable;
- high-variance activation PC direction;
- matched random direction orthogonal to protected output span.

Freeze a manageable count, such as 8 directions per cell, before Stage 2.

## 3.9 Epsilon ladder

Start with relative scales:

```text
{0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20} × ||h||
```

The usable ladder is the subset that passes delivery and response-SNR gates. The earlier handout’s `{0.01,0.02,0.05,0.10,0.20}` remains the core. Smaller points help identify intercept but may be below bf16 delivery.

For each epsilon, run `-epsilon`, `+epsilon`, and `2epsilon`. Pairwise additivity uses a separate fixed direction pair.

## 3.10 OLMo positive control

Before interpreting Gemma, run the identical exact-JVP harness on OLMo 3.1 Think or the exact positive-control checkpoint already used by the campaign.

Acceptance:

- exact JVP matches smallest faithful secant;
- tangent error grows smoothly with epsilon;
- known linear band passes more strongly than shallow controls;
- single-position and uniform differences reproduce the earlier qualitative result;
- no unexplained path mismatch.

If OLMo fails, debug the harness. Do not conclude both models are nonlinear.

## 3.11 G1 evidence IDs

```text
gm-jvp-harness-goldens-v1
gm-jvp-olmo-positive-control-v1
gm-jvp-gemma-stage1-v1
gm-jvp-gemma-stage2-v1
```

## 3.12 G1 branch rules

| Result | Interpretation | Next step |
|---|---|---|
| JVP fails to match tiny faithful secant on both models | harness/path bug | stop and repair |
| OLMo passes, Gemma tiny secant fails | Gemma path or nondifferentiable implementation issue | unfused/parity audit |
| Gemma tiny secant matches, error grows with epsilon | genuine finite curvature | G2/G3/G4 |
| prompt-specific JVP passes finite scale, mean fitted J fails | context/position averaging | G5 |
| late layers pass while mid-band fails | relocated transport regime | G6 |
| all faithful cells pass | prior nonlinearity claim superseded | audit old harness and re-evaluate mean-map fit |

---

# 4. G2: layerwise and sublayer localization

## 4.1 Cumulative response surface

For one frozen source perturbation, capture clean, JVP, and secant responses after every downstream stage:

```text
block input
pre-attention norm
Q/K/V projections or equivalent internal checkpoint
attention probabilities/logits
attention output before projection
attention branch output
post-attention norm if present
residual add
pre-MLP norm
MLP gate preactivation
MLP value branch
gated product
MLP output
post-MLP norm if present
residual add
```

Adapt names to the exact model implementation. The architecture manifest defines the valid stages.

## 4.2 Cumulative versus incremental defect

For stage `k`, compute cumulative tangent error `E_k`. Then report the change across the stage:

```text
jump_k = E_k - E_{k-1}
```

Also compute a local stage secant/JVP from the clean stage input to stage output where possible. A cumulative jump can arise from amplifying an earlier error; a local defect shows the stage itself is nonlinear.

## 4.3 Block-type matched analysis

Gemma alternates local and full-attention blocks. Test whether defect jumps concentrate at full-attention blocks while controlling for depth.

Use:

- matched local block immediately before each full block;
- matched local block immediately after;
- normalized distance from source and target;
- prompt as repeated unit;
- family/prompt bootstrap;
- full-versus-local contrast over repeated six-block groups.

The unit of inference should be prompt × block group, not individual hidden dimensions.

## 4.4 Layer transition around L42-L44

Because answer ranks collapse there, increase resolution around:

```text
L38, L39, L40, L41, L42, L43, L44, L45, L46, L48, L52
```

Measure:

- output-basis rank;
- exact tangent fidelity;
- finite curvature slope;
- attention route entropy;
- representation similarity between adjacent layers;
- norm and radial/tangent decomposition;
- fitted mean-map identification;
- selected-token direction emergence.

Test whether the readout transition aligns with a transport-validity transition, a representation rotation, or merely increasing proximity to the output.

## 4.5 Evidence

```text
gm-layerwise-response-localization-v1
gm-sublayer-defect-localization-v1
gm-local-vs-full-attention-contrast-v1
gm-l42-l44-transition-v1
```

---

# 5. G3: attention-routing autopsy

## 5.1 Hypothesis

Finite perturbations may change attention routes. Gemma’s repeated local blocks punctuated by full-attention blocks may concentrate long-range state changes into a few routing transitions. The attention Jacobian includes value-path and routing terms, plus QKNorm derivatives and q·k cross effects.

## 5.2 Required clean caches

For each prompt/block/head:

- normalized input;
- Q, K, V;
- QKNorm outputs;
- RoPE-transformed Q/K;
- attention logits;
- mask;
- attention probabilities;
- attention output before and after output projection;
- block type and window/global routing metadata.

## 5.3 Routing variants

Implement one block at a time with clean parity tests.

### R0: full live attention

Original model.

### R1: clean attention probabilities frozen, values live

Use cached clean `A_0` and recomputed `V(h+δ)`:

```text
O = A_0 V(h+δ)
```

This removes routing curvature while retaining the value path.

### R2: Q/K route frozen, values live

Cache normalized/RoPE Q and K or the resulting logits, depending on the implementation. Recompute V. This should agree closely with R1 unless other dropout/masking details intervene.

### R3: values frozen, routing live

Use live attention probabilities with clean V. This isolates route movement.

### R4: query-only or key-only live

Where tractable, freeze K and V while Q responds, then freeze Q and V while K responds. This separates source-token query changes from destination/key changes.

### R5: QKNorm linearized

Replace QKNorm with its first-order expansion around the clean Q/K state while keeping the rest live.

## 5.4 Tests

- clean no-perturbation parity for every variant;
- output parity with full attention when all live;
- gradient/JVP parity for R0;
- exact probability row sums;
- mask and local-window parity;
- no accidental cross-token information under local blocks;
- global `V=K` behavior preserved if the exact checkpoint implements it;
- source and target positions logged.

## 5.5 Decision logic

- R1/R2 substantially restore tangent accuracy: routing movement is the main curvature source.
- R3 carries most defect: routing-only response dominates.
- effect concentrated in full-attention blocks: global routing events are load-bearing.
- local blocks equally defective: repeated local route or norm/gate effects dominate.
- QKNorm linearization restores accuracy: normalization inside attention is central.
- no routing variant helps: proceed to norm/MLP and context analyses.

## 5.6 Causal relevance

Run routing freezes only as transport diagnostics on known/development prompts. Do not claim that freezing attention is a valid workspace intervention. It changes the model mechanism and exists to localize curvature.

---

# 6. G4: normalization and gated-MLP autopsy

## 6.1 RMSNorm first-order replacement

For clean input `x0`, define:

```text
N_lin(x) = N(x0) + D N(x0) (x - x0)
```

Implement an analytic RMSNorm JVP or use exact autodiff to construct the local linear action without materializing a full d×d Jacobian.

Variants:

- linearize only pre-attention norm;
- only post-attention norm;
- only pre-MLP norm;
- only post-MLP norm;
- all downstream RMSNorms from one source layer;
- radial versus tangent perturbations.

If curvature is primarily radial, tangent directions should improve strongly.

## 6.2 QKNorm

Treat QKNorm separately from residual RMSNorm. Compare:

- live QKNorm;
- frozen clean QKNorm output plus first-order value changes where meaningful;
- first-order QKNorm;
- no QKNorm only as a non-scientific diagnostic, not a parity-preserving model.

## 6.3 Gated MLP decomposition

For a simplified gated MLP:

```text
g(x) = phi(W_g x) ⊙ (W_v x)
```

At clean `x0`, the first-order response is:

```text
Dg(x0)δ = [phi'(W_g x0) ⊙ (W_v x0)] ⊙ (W_g δ)
          + phi(W_g x0) ⊙ (W_v δ)
```

Implement diagnostic variants:

### M0: full live gate/value

Original MLP.

### M1: gate frozen, value live

```text
phi(W_g x0) ⊙ W_v x
```

### M2: gate live, value frozen

```text
phi(W_g x) ⊙ W_v x0
```

### M3: first-order gate, value live to first order

Use the local Taylor expansion.

### M4: full first-order MLP

Replace the complete MLP response by its clean affine linearization.

## 6.4 Factorial localization

On the strongest Gemma failure cells, run:

```text
original
routing frozen
norms linearized
MLP linearized
routing + norms
routing + MLP
norms + MLP
all three
```

Use a small frozen prompt/direction set. This shows whether defects add independently or arise from interactions.

## 6.5 Decision logic

- norms alone restore tangent accuracy: repeated normalization curvature;
- MLP linearization restores additivity: multiplicative gate cross-curvature;
- combined restoration much larger than either: interaction across stages;
- none restore: context heterogeneity, source hook semantics, or other architecture-specific operations remain.

---

# 7. G5: prompt-specific tangents versus average-map cancellation

## 7.1 Core separation

For each prompt/direction:

1. exact prompt-specific JVP `J_x v`;
2. fitted corpus-average `J_bar v` from the existing lens;
3. mean exact JVP over a training prompt set;
4. cluster-conditioned mean JVP;
5. finite secant at the assay scale.

This produces two independent errors:

```text
local curvature error: Delta_x(epsilon) vs epsilon J_x v
mean-map error:        J_x v vs J_bar v
```

Do not infer one from the other.

## 7.2 Fixed directional probe basis

A full prompt-specific d×d Jacobian is unnecessary and expensive. Freeze a probe basis of 32-128 directions per layer:

- Rademacher;
- random Gaussian;
- activation PCs;
- J-selected token directions;
- answer/bridge directions;
- radial/tangent directions.

Collect `Y_x` response matrices.

## 7.3 Context features for clustering

Use only features available without target outcome leakage:

- prompt length;
- task type;
- local/full attention route entropy;
- source activation norm;
- answer/bridge token presence;
- hidden-state PCA coordinates;
- attention-pattern summaries;
- layer/block type.

Do not cluster on tangent error itself and then claim prediction.

## 7.4 Cross-validation

Split by prompt family. Fit clusters or a gating model on training prompts, then evaluate:

- tangent-direction prediction on held-out prompts;
- finite secant prediction at small/moderate epsilon;
- readout token ranking;
- whether clusters remain stable across random seeds.

Compare:

```text
one global mean map
K=2,4,8 cluster maps
prompt-specific JVP oracle
random cluster control
```

## 7.5 Outcomes

### Context heterogeneity

Prompt-specific JVP predicts finite responses well, global mean fails, cluster maps improve held-out prediction.

### True curvature

Prompt-specific JVP already fails at moderate or tiny faithful scales; clusters cannot rescue finite response.

### Mixed

Prompt-specific tangent works at tiny scale, cluster maps improve mean prediction, but finite-dose error still grows.

## 7.6 Mean-lens identification at L22

The prior four-slice mean maps are near-orthogonal at L22. G5 should determine whether:

- prompt-specific tangents themselves vary strongly;
- the tangent signal is near zero and dominated by estimation noise;
- contexts divide into stable but opposing charts;
- one prompt stratum creates cancellation.

No readout or causal claim at L22 is licensed unless a valid context-conditioned instrument emerges.

---

# 8. G6: late-band transport and workspace-readout assay

## 8.1 Entry criterion

Run a J-space-style late-band assay only at layers/scales where:

- faithful perturbation delivery passes;
- exact JVP matches tiny secant;
- finite tangent error is within a control-calibrated acceptance region at the intended intervention dose or a smaller declared dose;
- mean or cluster map is stable enough for the proposed readout/intervention;
- positive controls pass.

Likely candidate band:

```text
L44, L48, L52
```

but the exact JVP results decide.

## 8.2 Late-band readout

Measure:

- accepted answer rank;
- bridge/intermediate rank;
- J-over-logit gain;
- capacity and occupancy;
- persistence across layers and positions;
- lead time to emitted tokens;
- competition among active directions;
- context-cluster dependence.

## 8.3 Output-staging versus workspace

Late readability alone is not workspace evidence. Distinguish by:

- lead time: does the direction precede emission by many tokens?
- persistence: does it survive across reasoning steps?
- capacity: is there sparse competition?
- broadcast: do multiple downstream components consume it?
- causal specificity: does span-safe removal affect multi-step reasoning beyond exact controls while protecting imminent output?
- prose guard: is damage selective or generic?
- phase: prefill/reasoning versus final-answer only.

## 8.4 Intervention scale

If only small perturbations pass the transport gate, use steering/readout or local derivative claims. Do not jump to a large orthogonal projection ablation. A valid tangent at epsilon 0.01 does not license deleting 20% of residual energy.

## 8.5 Possible conclusion

A late valid band may show:

- output staging only;
- a delayed verbalizable workspace;
- a context-specific readable chart;
- no sparse causal channel despite readout;
- a useful scoped transport instrument.

Each is publishable as a boundary result when stated precisely.

---

# 9. G7: nonlinear transport recovery

## 9.1 Path-integrated Jacobian

For a specified intervention path `h+sδ`:

```text
F(h+δ)-F(h) = ∫_0^1 D F(h+sδ) δ ds
```

Approximate with quadrature using K JVPs:

```text
K ∈ {1, 2, 4, 8, 16}
```

Compare left, midpoint, trapezoidal, and Gauss-Legendre rules if practical.

Metrics:

- response cosine;
- gain;
- relative error;
- compute cost;
- stability across prompts and directions;
- whether the path crosses attention-route transitions.

K=1 at s=0 is the ordinary tangent. Improvement with K supports finite curvature that can be integrated.

## 9.2 Conceptual limitation

Path-integrated transport is path and context dependent. A token direction becomes a vector field, not one fixed dictionary row. Projection ablation changes the path and therefore the derivative field. This method can predict a specified intervention response without automatically restoring the original global J-space interpretation.

## 9.3 Directional second order

Estimate:

```text
Delta(epsilon;v) ≈ epsilon Jv + 0.5 epsilon^2 H[v,v]
```

Use nested JVP or HVP machinery on a small direction set. Compare:

- first order;
- second order;
- path-integrated K=2/4;
- actual secant.

Report whether curvature is low-rank or concentrated in J-selected directions.

## 9.4 Bilinear cross terms

Estimate `H[v,w]` through polarization or mixed directional derivatives. Test whether MLP gate or routing curvature produces strong interactions specifically between answer and bridge directions.

## 9.5 Mixture of local linear maps

If G5 supports context clusters, fit a piecewise-linear atlas and evaluate held-out prediction. Keep the gate frozen and inspectable. Do not fit a large neural router that can memorize prompts.

## 9.6 Recovery branch

- path integration closes most finite error: publish a nonlinear-transport methods result;
- second order closes error with few HVPs: curvature is structured;
- cluster maps close mean error but not finite error: heterogeneity plus curvature;
- no method closes error economically: hard applicability boundary for J-space on this checkpoint.

---

# 10. G8: nonlinear probes, readout only

## 10.1 Purpose

A nonlinear probe can answer whether Gemma’s mid-band residual contains decodable answer or bridge information even when a linear output-coordinate map fails.

It cannot by itself show that the model uses the decoded feature.

## 10.2 Probe design

- small MLP and kernel baselines;
- matched parameter-count linear probe;
- train/validation/test split by relation family;
- prompt-template holdout;
- label permutation;
- length and surface-form controls;
- counterfactual bridge/answer labels;
- calibration and selectivity;
- no layer selection on the test set.

## 10.3 Cross-model comparison

Run the same probe budget on OLMo and optionally Qwen. A nonlinear probe advantage unique to Gemma supports a private nonlinear code. A similar advantage everywhere may merely reflect probe capacity.

## 10.4 Causal boundary

Do not define a nonlinear “concept removal” until:

- probe generalizes strongly;
- representation manifold constraints are defined;
- matched controls exist;
- a layer has a valid downstream transport regime;
- intervention optimization does not leave the activation manifold.

Nonlinear causal removal is future work unless the simpler transport autopsy closes cleanly.

---

# 11. Architecture-comparison extensions

## 11.1 Within-Gemma size/checkpoint comparison

Only after the 31B mechanism is localized, consider another Gemma checkpoint or size. The comparison must ask a specific prediction:

- same local/global attention schedule but different width;
- same size, base versus instruct;
- Gemma 3 versus Gemma 4 architecture change;
- per-layer input feature enabled versus disabled if a valid matched model exists.

Do not add models merely to create a colorful table.

## 11.2 Positive controls

At minimum OLMo. Optional Qwen can test whether hybrid/linear-attention architecture produces a third transport regime, but it should not delay Gemma closure.

## 11.3 Matched relative depth and distance-to-target

Cross-model comparisons must match both:

- relative depth;
- remaining number of downstream blocks.

The latter matters because near-target maps become identity-like. Report identity-subtracted operator metrics.

---

# 12. Statistical plan

## 12.1 Units

- prompt is the primary resampling unit for transport validity;
- block group is a repeated measure within prompt;
- direction is a designed repeated measure, not an independent sample;
- relation family is the unit for nonlinear-probe generalization;
- model is not a population random effect with only two or three models.

## 12.2 Stage separation

- Stage 1 harness validation: tiny models and 4-prompt gates;
- Stage 2 development: frozen 16+ prompt set;
- Stage 3 validation: independent prompt/family set for the selected mechanism;
- do not choose the mechanism and validate it on the same prompt set.

## 12.3 Intervals

- prompt bootstrap for aggregate metrics;
- paired bootstrap for original versus linearized/frozen variants;
- block-group cluster bootstrap for full versus local attention;
- robust regression interval for `a+b epsilon`;
- family-held-out confidence for probes;
- report full distributions and failure rates, not only medians.

## 12.4 Multiplicity

Define a small set of primary methods questions:

1. exact tangent validity in mid-band;
2. full-versus-local attention defect contrast;
3. best precommitted mechanism restoration contrast;
4. late-band transport validity.

Treat dense layer/direction surfaces as exploratory with FDR or descriptive intervals. Do not Holm-correct hundreds of hidden-dimension cells.

## 12.5 Acceptance regions

Thresholds must be calibrated against:

- OLMo positive control;
- random directions;
- J-selected intervention directions;
- intended finite dose;
- clean and wrong-hook sentinels;
- smallest faithful epsilon.

No universal cosine 0.9 theorem exists. Freeze operational thresholds before Stage 2 target outcomes.

---

# 13. Stop rules and incident handling

## 13.1 Hard stop conditions

Stop before interpretation for:

- wrong model or processor revision;
- architecture manifest mismatch;
- hook clean-parity failure;
- functional suffix mismatch with full forward;
- exact JVP backend silently falling back to detached or finite-difference behavior;
- OLMo positive control failure;
- perturbation delivery below fidelity threshold;
- response below frozen SNR floor;
- wrong source or target position;
- cache/no-cache drift;
- fused/eager replacement clean parity failure;
- malformed attention mask/window behavior;
- dirty producer tree;
- output path reuse;
- checkpoint/input-manifest mismatch.

## 13.2 Incident record

Every failed attempt writes an immutable diagnostic event containing:

- command;
- code commit;
- environment;
- failure stage;
- no scientific result if the gate failed;
- output paths and hashes;
- supersession relation when repaired.

Do not delete failed artifacts that explain a later correction.

## 13.3 Contradiction heuristic

The campaign’s standing rule remains useful:

> When a new instrument contradicts an established result, suspect the instrument first and design a discriminating check.

Examples already caught include wrong estimands, invalid family clustering, output-span leakage, bf16 floors, and mean-map identification failures. Apply this rule without using it to dismiss inconvenient validated results.

---

# 14. First 24-hour execution plan

## CPU lane, immediately

1. Create side package, registry, and foundation manifest.
2. Import and verify prior Gemma/OLMo evidence.
3. Build exact architecture and block-type manifest.
4. Write tiny differentiable transformer goldens with known analytic JVP/Hessian.
5. Implement JVP backend abstraction and tests.
6. Implement perturbation-delivery and target-extraction contracts.
7. Freeze Stage 1 prompts, layers, directions, epsilon ladder, and thresholds.
8. Prepare OLMo and Gemma configs.
9. Prepare report skeleton and decision table.

## GPU lane

### G1-A: backend smoke

- tiny-model exact JVP versus analytic answer;
- one small Hugging Face transformer suffix;
- one OLMo prompt/layer/direction;
- one Gemma late layer;
- verify no unsupported/custom op issue.

### G1-B: OLMo positive control

Run 4 prompts × matched layers × frozen directions × epsilon ladder. Register only after control thresholds pass.

### G1-C: Gemma Stage 1

Run 4 prompts at L22/L30/L37/L44/L52, single-position primary and uniform secondary, with random/radial/tangent/J-selected directions.

Apply branch rules before scaling.

### G1-D: Gemma Stage 2

If Stage 1 is valid, expand to 16 prompts and the full layer/direction set.

### G2

On the two strongest validated failure cells and one late passing/control cell, run layerwise/sublayer localization.

### G3, if time

Implement and run clean-probability-frozen attention on one local block and one matched full-attention block. This is the first mechanism discriminator.

## Never-drop items

- exact JVP backend and goldens;
- OLMo positive control;
- Gemma Stage 1 exact JVP/secant result;
- full provenance and raw rows;
- decision-tree classification.

## Drop order

```text
nonlinear probe
-> second-order HVP
-> path integration beyond K=4
-> full context clustering
-> broad routing factorial
-> Stage 2 prompt expansion
```

Do not drop the positive control or exact JVP to fit more Gemma layers.

---

# 15. Proposed code and evidence map

## 15.1 Core code

```text
jspace_gemma/autodiff.py
jspace_gemma/architecture.py
jspace_gemma/hooks.py
jspace_gemma/linearizations.py
jspace_gemma/targets.py
jspace_gemma/transport_metrics.py
```

## 15.2 Experiment modules

```text
experiments/gm_foundation.py
experiments/gm_jvp_goldens.py
experiments/gm_exact_transport_gate.py
experiments/gm_layerwise_localization.py
experiments/gm_attention_routing_freeze.py
experiments/gm_norm_mlp_linearization.py
experiments/gm_context_tangent_atlas.py
experiments/gm_late_band_assay.py
experiments/gm_path_integrated_transport.py
experiments/gm_directional_second_order.py
experiments/gm_nonlinear_probe.py
experiments/gm_release_manifest.py
```

## 15.3 Evidence IDs

```text
gm-foundation-v1
gm-jvp-goldens-v1
gm-jvp-olmo-positive-control-v1
gm-jvp-gemma-stage1-v1
gm-jvp-gemma-stage2-v1
gm-layerwise-localization-v1
gm-attention-routing-freeze-v1
gm-norm-mlp-linearization-v1
gm-context-tangent-atlas-v1
gm-late-band-transport-v1
gm-path-integrated-transport-v1
gm-second-order-transport-v1
gm-nonlinear-readout-v1
gm-state-of-record-v1
```

Use new versions for repairs. Never reuse an evidence ID.

---

# 16. Required figures

1. **Exact tangent error by epsilon and layer**
   - Gemma and OLMo;
   - `a+b epsilon` fits;
   - single-position versus uniform.

2. **Decision decomposition**
   - prompt-specific JVP error;
   - mean-map error;
   - finite-dose curvature.

3. **Layer/sublayer defect waterfall**
   - cumulative and local error;
   - local versus full-attention blocks.

4. **Routing intervention panel**
   - full live;
   - frozen route/live value;
   - live route/frozen value;
   - QKNorm linearized.

5. **Norm/MLP restoration panel**
   - one-at-a-time and combined variants.

6. **Context atlas**
   - prompt-map similarity;
   - global versus cluster-held-out prediction.

7. **Late-band validity/readout panel**
   - rank, tangent fidelity, capacity, lead, and causal scope.

8. **Methods decision tree with observed branch highlighted**

Every figure must read registered tables. No live-tensor plotting.

---

# 17. Claim ladder and branch conclusions

## Branch G-A: numerical or implementation artifact

Criteria:

- exact JVP fails against the smallest faithful secant;
- result changes qualitatively under parity-preserving implementation or dtype repair;
- OLMo control identifies the same issue.

Licensed conclusion:

> The prior Gemma nonlinearity result was not established; the measurement path was invalid at the tested cells.

Then rerun the repaired gate before any mechanism claim.

## Branch G-B: finite within-context curvature

Criteria:

- exact JVP matches the smallest faithful secant;
- error grows with epsilon;
- prompt-specific tangent already fails at assay scale;
- context clustering does not remove the finite-dose defect.

Licensed conclusion:

> Gemma’s mid-band downstream map is differentiable but has a narrow tangent regime relative to the projection-scale J-space intervention.

## Branch G-C: mean-map cancellation

Criteria:

- prompt-specific tangent predicts small/moderate secants;
- global mean map fails;
- context clusters improve held-out prediction;
- slice cancellation and L22 instability are explained.

Licensed conclusion:

> Gemma uses context-dependent local transport charts that do not average to one useful fixed mid-band J-lens.

## Branch G-D: routing curvature

Criteria:

- defect jumps at attention stages or full-attention blocks;
- frozen clean routes/live values restore tangent accuracy;
- route-only variants carry the defect.

Licensed conclusion:

> Context-dependent attention routing is the dominant source of finite transport curvature in the tested band.

## Branch G-E: normalization or gated-MLP curvature

Criteria:

- local linearization of norms or gate/value product substantially restores additivity and tangent prediction.

Licensed conclusion names only the operations that pass the discriminating test.

## Branch G-F: late valid band

Criteria:

- exact transport passes at L44-L52;
- mean/cluster map stable;
- readout and causal tests are valid at a declared dose.

Licensed conclusion:

> A useful J-style transport instrument exists only in a later Gemma band, whose content must be tested for output staging versus workspace-like persistence and causality.

## Branch G-G: nonlinear recovery

Criteria:

- path-integrated or second-order transport predicts finite responses materially better on held-out prompts.

Licensed conclusion:

> Gemma’s transport is recoverable for specified paths with a context-dependent nonlinear extension, but not as one fixed global token dictionary.

## Branch G-H: hard methods boundary

Criteria:

- exact JVP confirms curvature/heterogeneity;
- routing/norm/MLP localization remains mixed;
- practical nonlinear recovery fails or is too expensive;
- no late valid band supports the intended assay.

Licensed conclusion:

> The fixed J-space method has a checkpoint-specific applicability boundary on Gemma 4 31B at the tested depths and doses.

This is a complete result, not a failed experiment.

---

# 18. Reproduction and release

## 18.1 Raw row schema

Every transport row includes:

```text
prompt/family id and hash
model/tokenizer/processor revision
source layer, source position, perturbation mode
target stage and target representation
direction type/id/hash
desired and realized epsilon/direction
input fidelity, response SNR
exact JVP backend and implementation hash
secant outputs at -1,+1,+2
tangent cosine, gain, relative error
homogeneity, odd symmetry, additivity
block type and local/global metadata
routing/norm/MLP variant
code/config/environment hashes
```

## 18.2 Independent reproduction

- tiny analytic goldens in a fresh process;
- one complete OLMo cell;
- one complete Gemma Stage 1 cell;
- recompute all metrics from raw response vectors;
- rerun regression fits;
- regenerate figures;
- verify no report code reads expected conclusions;
- hash all outputs.

## 18.3 Release bundle

```text
release/GEMMA_TRANSPORT_STATE_OF_RECORD.md
release/IMPORT_BUNDLE_PHASE4.json
release/IMPORT_BUNDLE_PHASE4.md
release/gemma_transport_inventory.json
release/gemma_transport_environment_lock.json
release/gemma_transport_claim_ledger.md
```

The Phase 4 import bundle should expose only methods conclusions and any validated transport gate. It must not turn this side study into a Phase 4 confirmatory model cell.

---

# 19. Completion criteria

The first Gemma side study is complete when:

- [ ] isolated package, registry, run root, and import manifest are verified;
- [ ] architecture and hook manifest is complete;
- [ ] exact JVP backend passes analytic and transformer goldens;
- [ ] OLMo positive control passes;
- [ ] Gemma Stage 1 and Stage 2 exact JVP/secant results are registered;
- [ ] finite curvature versus mean-map cancellation is classified;
- [ ] layer and sublayer defect localization is complete on validation prompts;
- [ ] at least one routing discriminator and one norm/MLP discriminator are tested;
- [ ] late L44-L52 band is explicitly licensed or rejected for scoped transport;
- [ ] path-integrated or second-order recovery is tested if finite curvature is confirmed;
- [ ] nonlinear probe is clearly separated from causal evidence;
- [ ] every claim carries a layer, position, direction, target, epsilon, and tier scope;
- [ ] figures regenerate from registered rows;
- [ ] independent reproduction passes;
- [ ] state-of-record and Phase 4 import bundle are complete.

At that boundary, stop. The next phase should be chosen from the observed branch: routing mechanism, context atlas, nonlinear transport method, late-band workspace test, or a clean applicability-boundary paper. Do not keep adding Gemma variants until the 31B mechanism is understood.
