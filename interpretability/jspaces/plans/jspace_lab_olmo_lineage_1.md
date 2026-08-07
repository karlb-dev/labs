# jspace_lab_olmo_lineage_1.md

## OLMo lineage side study 1: what post-training changes in J-space geometry, causal use, and workspace-like organization

**Purpose:** run a dedicated, parallel OLMo program that answers the open questions in `olmo_lineage_revised.tex` without blocking or contaminating the main Phase 4 freeze. The scientific center is the OLMo 32B lineage. Qwen and Gemma remain comparison points, not co-equal model sweeps.

**Starting boundary:** fork from the clean Phase 4 branch boundary `3b041735d8b842de46a9c0a474fccd0c44e0841a`. Import the registered Phase 3 and Phase 4 OLMo artifacts by hash. Do not write to the Phase 4 registry, run root, preregistration, or reports. Do not open any untouched Phase 4 confirmatory or replication intervention outcome. The side study may use known Phase 3 banks and the Bank W **development** partition. It must label every result development or methods tier.

**Recommended branch and namespace:**

```text
branch:   interp_jspace_olmo_lineage
package:  interpretability/jspace_olmo_lineage/
run root: /content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_<date>/
registry: interpretability/jspace_olmo_lineage/reports/evidence_events.jsonl
prefix:   ol-
```

> **Paste-line for the coding/research agent**
>
> Read `olmo_lineage_revised.tex` and PDF, `PHASE3_STATE_OF_RECORD.md`, `PHASE4_DEVELOPMENT_REPORT.md`, the Phase 4 candidate preregistration, the latest Phase 4 handoff, and the accepted 4.1/4.2 plans before writing code. This is a parallel development study, not a new confirmatory phase. First create an isolated `jspace_olmo_lineage` package, run root, registry, and immutable import manifest from the exact Phase 4 boundary. Do not edit `jspace_phase4` or its registry. Immediately run the two Phase-4-compatible OLMo Bank-W baseline capability gates and produce a hash-pinned import bundle for the main branch. Then use only the known/development families to answer the lineage questions in this plan: measure base capacity with the corrected paper-defined estimator; audit and, only if necessary, refit a same-corpus lens series across Base, 3.0 Think, 3.1 Think, and 3.1 Instruct; separate activation state, lens coordinates, output readout, and downstream receiver effects through a crossed model/lens analysis; run the Bank-W load × derivation × redundancy development grid; model accessibility and answer commitment continuously; and localize downstream receiver components with within-model rescue and cross-checkpoint patching. Every result must retain the matched-lineage natural-experiment limitation. Never claim that Think training creates a workspace unless a later randomized or controlled-training study earns that wording. Commit and push at each evidence boundary, checkpoint GPU work to the side-study Drive root, and finish with `IMPORT_BUNDLE_PHASE4.json` plus a paper-facing claim ledger.

---

# 0. Executive thesis and decision frame

## 0.1 What the existing lineage already shows

The current OLMo development trajectory is unusually coherent:

- the pretrained Base checkpoint has approximately zero span-safe J-specific effect in every primary known-bank cell;
- Bank-S direct dependence appears by the released OLMo-3 32B Think checkpoint;
- the direct effect remains at OLMo-3.1 32B Think;
- the composed-minus-direct contrast becomes clearly positive at 3.1 Think;
- the OLMo-3.1 32B Instruct sibling returns near zero on the same Bank-S quantities;
- the pattern survives both each checkpoint’s own J-lens and a frozen base-lens coordinate frame;
- the exact common-support analysis preserves the Base-to-Think direct decrease and the Think-to-Instruct reversal on the same facts;
- the exact rank-and-energy control remains near zero while the span-safe J arm moves;
- measured centered excess and occupancy are nearly unchanged across the three post-trained checkpoints currently measured.

The most economical present interpretation is:

> Post-training changes the recruitment, routing, or downstream use of a thin J-readable channel more than it changes the channel’s measured sparse capacity.

That sentence is development-tier and appropriately cautious. It leaves five distinct mechanisms unresolved:

1. the dictionary may already exist at Base and be recruited by Think training;
2. the dictionary itself may rotate or sharpen while aggregate effects survive a common frame;
3. Think may externalize state into the prompt or reasoning stream, reducing dependence on the internal channel for composed prompts;
4. Instruct may route task state closer to output commitment or through different receivers;
5. the apparent transition may arise at one unobserved stage in a long post-training recipe.

This side study is designed to discriminate those mechanisms rather than accumulate another stack of checkpoint means.

## 0.2 “Workspace formation” must be decomposed

Do not use one scalar “workspace score.” Track at least six axes:

### A. Coordinate availability

Does a stable token-readable transport dictionary exist at the checkpoint and layer?

### B. Sparse capacity

How many directions are occupied under the paper-defined marginal-gain crossing, and what centered excess variance do they explain beyond a matched random dictionary?

### C. Causal utilization

Does removing the selected content damage behavior beyond an exact rank-and-energy-matched control?

### D. Downstream consumption

Which later attention or MLP components read the selected content, and can clean activation patching rescue the lesion?

### E. External-state substitution

Does prompt-supplied, repeated, or derived state reduce reliance on the internal channel?

### F. Temporal organization

When during prefill, reasoning, and final-answer formation does the channel become load-bearing?

A claim that post-training “forms a workspace” would require coordinated movement on several of these axes. The current evidence primarily shows movement on C, with hints about E, while B appears nearly flat across the post-trained trio. This plan tests A, B, D, E, and F directly.

## 0.3 The study is a matched-lineage natural experiment

The released graph is:

```text
                          unobserved post-training stages
Base  ------------------------------------------------->  OLMo-3 32B Think
                                                            |
                                                            v
                                                       OLMo-3.1 Think

Base  ---------------- distinct unobserved recipe ------>  OLMo-3.1 Instruct
```

The 3.1 Instruct checkpoint is a sibling endpoint, not the temporal successor of 3.1 Think. Shared architecture, width, base lineage, task battery, and intervention machinery make the comparison scientifically valuable. They do not randomize post-training data, objective, formatting, or checkpoint selection.

Every report sentence must distinguish:

- **localization:** the effect is visible between released endpoints;
- **association:** the effect covaries with the Think branch;
- **mechanism:** a measured geometry or receiver change explains the effect;
- **causation by training objective:** requires controlled training or a more isolated intervention.

The first three are achievable here. The fourth is optional and substantially harder.

## 0.4 Priority order

The recommended order is:

```text
O0  isolated foundation and immutable imports
O1  Phase-4-compatible Bank-W capability gates for the 3.1 pair
O2  corrected capacity on Base and a symmetric four-checkpoint capacity table
O3  same-corpus lens provenance audit, then minimal missing refits
O4  Bank-W development intervention grid across the lineage
O5  crossed activation × lens × readout analysis
O6  continuous accessibility and commitment analysis
O7  downstream receiver localization and patch rescue
O8  phase-resolved generation interventions
O9  intermediate-checkpoint or controlled-continuation study
O10 synthesis, independent reconstruction, and Phase 4 import bundle
```

For the first 24-hour VM, O0 through O4 are the core. O5 can begin if model and lens artifacts are already local. O7 through O9 should not displace the capability, capacity, and Bank-W work.

---

# 1. Foundation, governance, and imports

## 1.1 Create the package

Minimum tree:

```text
interpretability/jspace_olmo_lineage/
├── README.md
├── pyproject.toml
├── constraints.txt
├── configs/
├── data/
├── jspace_olmo_lineage/
│   ├── __init__.py
│   ├── __main__.py
│   ├── imports.py
│   ├── manifests.py
│   ├── paths.py
│   ├── provenance.py
│   ├── registry.py
│   ├── repro.py
│   ├── scoring.py
│   ├── stats.py
│   └── experiments/
├── preregistration/
├── protocol/
├── reports/
│   ├── evidence_events.jsonl
│   └── figures/
├── release/
├── reviews/
└── tests/
```

The side package may call stable public functions from Phase 4, but every imported source file and dependency revision must be recorded. Avoid copying large modules unless the side track needs to freeze a modified version.

## 1.2 Foundation evidence

Register:

```text
ol-foundation-v1
```

It should pin:

- parent repo commit;
- branch name;
- package source inventory;
- side run root;
- Python and package environment;
- CUDA gate and GPU identity;
- imported Phase 3 release manifest;
- imported Phase 4 registry boundary and selected evidence IDs;
- exact Bank F, Bank S, and Bank W hashes;
- exact model revisions, tokenizer revisions, lens hashes, and logical URIs;
- prohibition on Phase 4 untouched outcomes;
- test results.

## 1.3 Immutable import set

Import at minimum:

### Model and lens inputs

- OLMo-3-1125-32B Base revision and base lens;
- OLMo-3-32B Think revision and own lens;
- OLMo-3.1-32B Think revision and own lens;
- OLMo-3.1-32B Instruct revision and own lens;
- frozen base lens used as the common frame;
- every input manifest for the four existing lineage grids.

### Behavioral inputs

- Bank F v7;
- Bank S v3;
- G5 results for all four checkpoints;
- own/common lineage parquets and analyses;
- common-support population and contrast tables;
- Phase 3 span-audit and accessibility summaries;
- Phase 3 prose exact-control results.

### Phase 4 methods input

- Bank W candidate v2 and partition;
- Bank W capability protocol;
- Bank W power ruler;
- no Bank W intervention outcome, because none exists and none is licensed pre-freeze.

## 1.4 Import integrity tests

- exact hash match for every imported file;
- source event is live;
- source commit reachable;
- no side-track output path points into the source run root;
- imported parquets are opened read-only;
- family IDs and fact IDs match across G5, grids, and common-support tables;
- no Phase 4 confirmatory or replication family appears in a side-track development input;
- Base/Think/Instruct model labels cannot be inferred from position in a list alone;
- 3.1 Instruct is encoded as `sibling_endpoint`, never as the fourth temporal Think point.

---

# 2. Hypotheses and branch table

## 2.1 H1: conserved thin dictionary, changed recruitment

**Prediction:** same-corpus J operators and token-direction dictionaries are broadly aligned across checkpoints, sparse capacity is similar, but activation occupancy, selected rows, and downstream receiver dependence change along the Think path.

**Support already present:** common-base-lens effects preserve the Bank-S trajectory; post-trained capacity is nearly flat; matched dose stays near zero.

**Falsifier:** same-corpus J geometry changes so strongly that a fixed coordinate cannot represent the Think-path effect, or Base lacks comparable J-readable capacity entirely.

## 2.2 H2: external-state substitution on Think checkpoints

**Prediction:** on Think checkpoints, internally derived high-load state is most J-dependent; explicitly supplied or redundantly repeated state reduces J dependence. Instruct and Base show a weaker or absent derivation/redundancy interaction.

**Support already present:** Bank-S composed prompts are less J-dependent than direct prompts at Think, producing a positive composed-minus-direct contrast.

**Falsifier:** Bank W shows no derivation/redundancy moderation, or the effect is fully explained by baseline difficulty or prompt length.

## 2.3 H3: Instruct shifts state toward output-adjacent or alternative routes

**Prediction:** Instruct has comparable decodable state and capacity but different selected-span geometry, earlier answer commitment, different receiver components, or stronger alternate-route rescue.

**Support already present:** Instruct has near-zero Bank-S span-safe specificity while factual populations can still show large label-protected or span-safe effects; frame sensitivity appears in selected Bank-F cells.

**Falsifier:** receiver and commitment profiles are indistinguishable from Think after matching facts and geometry, with the observed Bank-S difference disappearing under better precision.

## 2.4 H4: accessibility gates causal cost

**Prediction:** within the OLMo lineage, J-specific damage is largest before an answer is fully committed, as measured by continuous accepted-answer mass, margin, entropy, and competing-answer structure, not merely clean rank.

**Support already present:** accessibility signs differ between OLMo and Qwen, and clean rank is coarse because many items sit at rank 1.

**Falsifier:** continuous commitment variables fail to predict within-family damage after controlling for task, model, layer, and dose.

## 2.5 H5: one unobserved stage installs the Think effect

**Prediction:** a matched intermediate checkpoint, if available, shows the transition localized to a narrower training interval. A controlled continuation emphasizing explicit reasoning produces movement in the same causal-utilization axis before major capacity change.

**Falsifier:** intermediate checkpoints show smooth drift or no monotone relation; a controlled continuation changes capability without reproducing the J-specific trajectory.

## 2.6 H6: transport validity itself changes across the lineage

**Prediction:** exact finite-dose transport gates pass across all OLMo checkpoints in the assay band, but fit estimation or context heterogeneity may differ. The causal trajectory should not be an artifact of one checkpoint having a more faithful J-lens.

**Falsifier:** one checkpoint fails the exact JVP/secant transport gate at the intervention scale or has dramatically worse mean-map fidelity that predicts the observed effect.

## 2.7 Outcome router

| Result pattern | Best interpretation | Next action |
|---|---|---|
| geometry and capacity stable, Bank W/receivers move | recruitment/routing change | mechanism paper spine |
| geometry moves, common-frame effects survive | coordinate drift plus stable coarse channel | report both axes; receiver tests decide |
| Base capacity materially lower | partial channel formation/growth | revise flat-capacity claim; test when growth occurs |
| Bank W derivation/redundancy interaction positive only on Think | external-state substitution | replicate on untouched Phase 4 families after freeze |
| Bank W null but accessibility explains damage | commitment-gated content channel | narrow workspace language |
| transport validity differs by checkpoint | instrument moderation | correct/limit lineage causal comparison |
| receiver patching rescues Think but not Instruct | downstream consumption difference | localize receiver path |
| all side-track mechanisms remain unresolved | retain development trajectory only | do not inflate causal story |

---

# 3. O1: Phase-4-compatible Bank-W capability gates

## 3.1 Immediate responsibility

Run the exact registered capability protocol on:

```text
OLMo-3.1 32B Think
OLMo-3.1 32B Instruct
```

These two outputs are required by main Phase 4. They must be generated first and without any intervention outcome.

Proposed configs:

```text
configs/ol_bank_w_capability_olmo31_think_dev.yaml
configs/ol_bank_w_capability_olmo31_instruct_dev.yaml
```

Proposed evidence IDs:

```text
ol-bank-w-capability-olmo31-think-dev-v1
ol-bank-w-capability-olmo31-instruct-dev-v1
```

## 3.2 Exact protocol

Use the registered Bank W **development** 24-family side, eight seeds per family, and only the primary capability cells:

```text
load 2, internally derived, stated once
load 6, internally derived, stated once
```

Score the full frozen eight-answer candidate set by summed sequence log probability with piecewise prompt/answer concatenation. Record:

- candidate scores;
- selected answer;
- true answer;
- accuracy;
- true-minus-best-wrong margin;
- prompt token count;
- answer token count;
- family, seed, load, superfamily;
- exact tokenizer IDs and manifest hash;
- GPU and model manifest;
- deterministic replay sentinel.

## 3.3 Acceptance rule

Per model:

- accuracy at load 2 >= 0.70;
- accuracy at load 6 >= 0.70;
- family-bootstrap 90% interval for high-minus-low accuracy wholly inside `[-0.08, +0.08]`;
- all 384 rows present and finite;
- no item dropped after scoring.

Then compute joint support across every independently eligible model, including the already registered Qwen gate:

- family-level accuracy >= 0.70 at both loads for every model in the set;
- at least 20 joint families.

Register:

```text
ol-bank-w-joint-support-dev-v1
```

This analysis contains no intervention column and may be imported into Phase 4.

## 3.4 Phase 4 import bundle, early release

As soon as O1 completes, emit a small early import bundle containing only:

- two capability results;
- two input manifests;
- joint-support result;
- source registry hash;
- package source hash;
- test summary.

Do not wait for the broader OLMo study before unblocking Phase 4.

---

# 4. O2: symmetric capacity across all four checkpoints

## 4.1 Why Base capacity is the first scientific hole

The current “capacity stays flat” statement is supported only across post-trained checkpoints for which the corrected estimator has been run. Base has a near-zero causal-utilization result, but its paper-defined sparse capacity is not yet measured under the exact same estimator and activation population.

Without Base capacity, three distinct stories remain open:

1. capacity exists at Base and Think recruits it;
2. capacity grows between Base and Think while occupancy remains small;
3. the existing post-trained capacity comparison is population- or fit-dependent.

## 4.2 Shared activation corpus

Freeze one activation corpus before measuring any checkpoint:

- 120 prompts, exact order and hashes;
- balanced factual prose, neutral prose, code/SQL, and in-context state templates;
- no Phase 4 untouched family content;
- same tokenizer-length inclusion rule adapted per model but a shared source text set;
- fixed layer set initially `{24, 32, 40}` plus optional normalized-depth neighbors;
- same position mask and skip-first rule;
- exact activation centering convention.

Prefer an existing registered activation corpus if it already meets these conditions. Do not author a new corpus solely to improve one model’s capacity.

## 4.3 Estimator

Use the corrected paper-defined procedure:

1. globally center activations over the frozen activation population;
2. perform nonnegative sparse pursuit in the J-token dictionary;
3. compare each marginal selected-direction gain with a matched-size random dictionary;
4. freeze occupancy at the crossing where marginal J improvement no longer exceeds the random baseline;
5. compute centered variance explained at that occupancy;
6. subtract matched random explained variance;
7. repeat random dictionaries under at least three stable seeds;
8. cluster uncertainty by activation prompt or frozen task family as appropriate.

Report separately:

- occupancy distribution;
- median occupancy;
- raw centered J share;
- random share;
- centered excess;
- uncentered/raw-energy sensitivity;
- solver residual and convergence;
- per-layer values;
- activation-corpus strata.

## 4.4 Four-checkpoint table

Run:

```text
Base
3.0 Think
3.1 Think
3.1 Instruct
```

Use the checkpoint’s own lens for the primary capacity table and the frozen base lens as a coordinate sensitivity. If an own lens is not same-corpus comparable, label the table accordingly and defer strong cross-checkpoint capacity language until O3.

## 4.5 Decision rules

Do not define a binary “capacity changed” threshold after seeing Base. Report paired or matched-corpus differences with intervals.

A useful development classification is:

- **stable:** absolute centered-excess difference < 0.25 percentage points and occupancy median unchanged, with equivalence interval inside a precommitted margin;
- **small shift:** 0.25 to 1.0 percentage points or one occupancy unit;
- **material shift:** >1 percentage point or >1 occupancy unit;
- **unresolved:** interval too wide for classification.

The margins should be declared in the capacity config before Base output is opened.

## 4.6 Additional capacity diagnostics

- capacity by direct/composed/supplied Bank W prompts on the development side;
- capacity by answer-commitment stratum;
- capacity under own versus common lens;
- occupancy overlap across checkpoints;
- relationship between selected support and causal effect, treated descriptively at item/family level;
- no model-level regression with four checkpoints presented as population inference.

---

# 5. O3: same-corpus lens geometry trajectory

## 5.1 Audit before refitting

The first action is a **lens provenance audit**, not four expensive refits.

For each existing lens, record:

- fitting corpus IDs, order, and hash;
- prompt count;
- maximum sequence length;
- skip-first rule;
- source and target layers;
- model dtype and accumulation dtype;
- `jlens` revision;
- tokenizer and BOS convention;
- slice/merge scheme;
- runtime package versions;
- final lens hash;
- independent-fit or slice evidence.

Classify pairwise comparability:

```text
EXACT_SAME_RECIPE_CORPUS
SAME_RECIPE_DIFFERENT_CORPUS
DIFFERENT_RECIPE
UNKNOWN
```

Only refit checkpoints necessary to create one exact same-corpus series.

## 5.2 Frozen corpus series

Use one leakage-safe nested corpus:

```text
OLMo draw A: n=120, optional n=250
```

The exact same 120 texts and order must be used at every checkpoint. The 250 extension is optional and should first be run on Base and 3.1 Think to estimate fit-size sensitivity. Do not launch four n=250 fits automatically.

## 5.3 Geometry metrics

At each source layer and for every checkpoint pair, report:

### Operator metrics

- raw matrix cosine;
- symmetric relative Frobenius delta;
- `J-I` and `J-alpha I` metrics;
- singular-value spectrum and effective rank;
- random transport-probe agreement;
- CKA on mapped token rows.

### Token dictionary metrics

- all-row or stable sampled-row cosine quantiles;
- task-token row cosines for Bank F, Bank S, and Bank W vocabularies;
- top-token neighbor overlap;
- row norm distributions;
- answer/bridge/shared-token strata.

### Sparse selection metrics

- selected-ID Jaccard;
- rank-biased overlap;
- projector overlap;
- principal angles;
- protected-span overlap;
- numerical-rank and near-tie rates;
- persistent direction overlap across positions.

### Readout components

- unembedding row cosine;
- final-norm gain cosine and scale;
- tied/untied status;
- plain logit-lens ranking;
- J-lens ranking;
- J-over-logit gain.

## 5.4 Geometry trajectory hypotheses

- H1 predicts broad base-to-Think operator and token-row continuity, with larger changes in selected activations and causal use than in the dictionary.
- A dictionary-formation account predicts large Base-to-3.0 geometry change and smaller 3.0-to-3.1 change.
- An Instruct-specific output-adjacent account predicts a sibling shift concentrated in late layers, answer directions, or norm/readout components.
- A pure lens-fit artifact predicts that same-corpus refits substantially reduce the observed trajectory.

## 5.5 Selection-margin audit

Reuse the Phase 4.3 top-k margin machinery. For each checkpoint:

- kth/k+1 score gaps;
- stable core versus near-tie fringe;
- alias-equivalent swaps;
- projector stability under fringe replacement;
- causal dose from core versus fringe.

This is especially important if the common-frame effect is stable but own-lens selected IDs differ.

## 5.6 Deliverable figure set

1. all-layer operator similarity heatmap;
2. assay-band token-row similarity by checkpoint edge;
3. selected-span overlap trajectory;
4. capacity versus causal-utilization state space;
5. unembedding/final-norm versus J-map movement decomposition.

Figures must be generated from registered tables, not directly from live tensors.

---

# 6. O4: Bank-W development intervention grid across the lineage

## 6.1 Why Bank W is the decisive OLMo experiment

The existing Bank-S result conflates several things:

- more than one inference step;
- explicit state in the prompt;
- repeated information;
- internal derivation;
- prompt length;
- task difficulty.

Bank W crosses the three mechanism variables explicitly:

```text
load:       2 vs 6 state elements
derivation: supplied vs internally derived
redundancy: stated once vs repeated
```

The primary Phase 4 cell is internally derived and stated once. This side study can use the full **development** factorial to explain the OLMo trajectory.

## 6.2 Models

Run development rows on:

```text
Base
3.0 Think
3.1 Think
3.1 Instruct
```

The Phase 4 primary only needs the 3.1 pair and Qwen after freeze. The side study gains scientific value by including Base and 3.0 Think on the development families.

## 6.3 Conditions

At minimum:

- baseline;
- span-safe J;
- exact instantaneous rank-and-energy matched control.

Optional development secondaries:

- label-protected J;
- protected-energy matched control;
- persistent matched control;
- logit-protected control.

Do not multiply conditions until the three core arms are complete on every model.

## 6.4 Primary side-study estimands

For model `m`, load `l`, derivation `d`, redundancy `r`:

```text
specific(m,l,d,r) = (LP_J - LP_0) - (LP_C - LP_0)
```

Key contrasts:

### Load engagement

```text
D_load = -[specific(high, derived, once)
           - specific(low, derived, once)]
```

Positive means high load is more J-dependent.

### External-state substitution

```text
D_supplied = specific(derived, once) - specific(supplied, once)
```

Interpret sign carefully under the negative-damage convention.

### Redundancy substitution

```text
D_repeat = specific(once) - specific(repeated)
```

### Three-way mechanism term

```text
D_mech = [high-low under derived-once]
         - [high-low under supplied/repeated control]
```

The exact contrast coding must be frozen in a config before intervention outcomes.

## 6.5 Analysis

- equal-family estimates and intervals;
- hierarchical model with family random intercept and fixed model/load/derivation/redundancy interactions;
- exact family sign flips for named paired contrasts;
- family leave-one-out;
- candidate-set accuracy and LP endpoints;
- baseline-capability covariates as frozen sensitivities, not exclusion after outcomes;
- no binary side-study claim from known development families.

## 6.6 Predicted patterns

### External-state substitution account

Think checkpoints:

- stronger negative J-specific effect at high internally derived load;
- reduced dependence when state is supplied;
- reduced dependence when state is repeated;
- effect already visible at 3.0 Think and not materially stronger at 3.1;
- Instruct and Base closer to zero or differently organized.

### Generic difficulty account

All models show more damage at high load, including matched control, and derivation/redundancy interactions disappear after baseline margin adjustment.

### Capacity account

Load slope tracks measured capacity rather than checkpoint role.

### Output-adjacent account

Damage concentrates near answer commitment and may appear in supplied/repeated cells despite low internal derivation demands.

## 6.7 Phase 4 import boundary

Only the 3.1 Think/Instruct **capability** results and joint support are imported before Phase 4 freeze. The broader development intervention results remain side-study evidence unless Phase 4 explicitly imports them after freeze as context. Never let known-family side outcomes influence untouched Phase 4 thresholds.

---

# 7. O5: crossed model, lens, readout, and activation analysis

## 7.1 The decomposition problem

A J-space effect at checkpoint `m` depends on at least four objects:

1. the activation `h_m` produced by the checkpoint;
2. the transport map `J_s` used to define token directions;
3. the final norm and unembedding `U_r` used to label/read those directions;
4. the downstream network of the checkpoint that consumes the ablated activation.

Existing own-lens and common-base-lens comparisons change item 2 while keeping activation and downstream model fixed and using the recipient readout convention. That is valuable but not a full decomposition.

## 7.2 Dictionary notation

For source transport checkpoint `s` and readout checkpoint `r`, define token direction matrix under one fixed convention:

```text
D_{s,r,l} = J_{s,l}^T U_r^T
```

or the row-equivalent implementation used by the project. Record the convention explicitly and test transpose parity.

Apply selection defined by `D_{s,r,l}` to activations from model `m`, then intervene in downstream model `m`.

## 7.3 Minimal crossed design

Do not run the full 4×4×4 cube immediately. Start with:

```text
activation/downstream model m ∈ {Base, 3.1 Think, 3.1 Instruct}
transport lens s            ∈ {Base, own(m)}
readout r                   ∈ {Base, own(m)}
```

Run on a balanced development subset of Bank-S direct/composed and Bank-W low/high derived-once families.

This isolates:

- activation/receiver change with a fixed dictionary;
- transport-map change with fixed activation/receiver;
- output-readout change with fixed transport;
- interactions.

## 7.4 Required controls

- exact rank/energy matched random for every crossed dictionary;
- protected-span geometry recomputed under each readout;
- same scientific seed namespace across compared frames;
- same item and condition order;
- numerical-rank-safe projectors;
- no comparing effects with unequal delivered rank/energy;
- logit-lens dictionary as a non-J baseline;
- wrong checkpoint labels blinded in the analysis table where practical.

## 7.5 Analysis model

At family level, estimate:

```text
specific ~ activation_model
         + transport_lens
         + readout
         + activation_model:transport_lens
         + activation_model:readout
         + task_cell
```

This is development estimation. Use paired contrasts that isolate one changed factor at a time. Report effect surfaces and uncertainty, not an omnibus ANOVA p-value as the main result.

## 7.6 Interpretation branches

- **Transport lens has little effect, activation model dominates:** training changes state occupancy or downstream use.
- **Transport lens explains most movement:** dictionary/map drift is load-bearing.
- **Readout explains movement:** final norm/unembedding or output adjacency is central.
- **Large interactions:** coordinate, state, and receiver co-adapt; simple conserved-dictionary language is too strong.

---

# 8. O6: accessibility and answer commitment

## 8.1 Replace rank with continuous commitment measures

Clean rank is too coarse when many accepted answers are rank 1. For every item and checkpoint, compute:

- logsumexp probability mass over accepted aliases;
- first-token accepted-answer mass;
- full-answer sequence LP;
- accepted-answer margin over best wrong candidate;
- entropy over a frozen candidate set;
- top-1/top-2 logit gap;
- rank and mass of bridge tokens;
- answer-direction projection in residual space;
- calibration or confidence under paraphrases;
- prompt-only versus bridge-supplied commitment.

## 8.2 Common-support population

Use the all-four and adjacent-pair common-support populations already registered. Do not condition on intervention outcomes.

## 8.3 Statistical model

A suitable development model is:

```text
specific_effect_{i,m}
  = beta_0
  + beta_model[m]
  + f(answer_margin_{i,m})
  + beta_entropy * entropy_{i,m}
  + beta_variant * composed_i
  + interactions
  + family_random_intercept
  + error
```

Use splines or rank-preserving bins for margin only if chosen before looking at lesion outcomes. Cross-validate predictions by family.

## 8.4 Matched-difficulty analysis

Create nearest-neighbor or coarsened exact matches across checkpoints using only baseline variables:

- task family;
- direct/composed;
- answer length;
- prompt length;
- accepted-answer margin;
- candidate entropy;
- capability status.

Then recompute checkpoint contrasts. This does not randomize training, but it tests whether accessibility distribution explains the trajectory.

## 8.5 Commitment-time experiment

For Think generation, track accepted-answer mass and J-specific sensitivity at:

- end of prefill;
- early reasoning;
- first point answer reaches rank 10;
- first point answer reaches rank 1;
- reasoning closure;
- first final-answer token.

Freeze event definitions from clean generation before intervention outcomes. Test whether lesion cost declines after commitment.

For Instruct, use prefill and final-answer stages; do not invent a reasoning phase.

---

# 9. O7: downstream receiver localization

## 9.1 Question

Does Think training change the J dictionary, or does it change the downstream components that consume an already available representation?

The most direct test is a lesion-rescue and receiver-localization study.

## 9.2 Within-model rescue, mandatory positive control

On 3.1 Think development families with strong Bank-S direct effects:

1. run clean and span-safe-ablated forwards;
2. cache clean residuals and component outputs;
3. patch the clean selected-J component back into the ablated run at candidate layers/positions;
4. measure answer LP recovery;
5. compare with random-subspace, wrong-layer, unrelated-item, and equal-energy patches.

Define recovery:

```text
recovery = (LP_patch - LP_ablated) / (LP_clean - LP_ablated)
```

Report raw LP changes too. Ratios can explode when the denominator is small.

## 9.3 Component search

At downstream layers, patch or ablate:

- attention outputs;
- MLP outputs;
- selected attention heads where feasible;
- residual stream before and after each sublayer.

Use a two-stage design:

1. broad screen on consumed development families;
2. held-out development-family validation of the top components.

Do not select and validate on the same families.

## 9.4 Path mediation

For candidate receiver `c`:

- lesion source J component;
- patch clean receiver activation;
- lesion receiver alone;
- patch unrelated receiver;
- patch wrong layer;
- patch matched random component;
- test whether source lesion effect is mediated by receiver restoration.

A receiver claim needs both necessity and rescue, not only attribution.

## 9.5 Cross-checkpoint patching

Raw residual patching across models can be off-manifold. Use staged tests:

### Stage 1: common-coordinate component patch

Estimate an orthogonal Procrustes alignment `R_{donor→recipient}` on a neutral, held-out corpus. Patch only the common J-span component:

```text
h_recipient' = h_recipient
             + lambda P_common (R h_donor - h_recipient)
```

### Stage 2: delta patch

Patch the clean-minus-ablated donor delta after alignment rather than the full state.

### Stage 3: receiver output patch

Patch the output of a homologous receiver component.

Controls:

- alignment fitted on disjoint prompts;
- random orthogonal alignment;
- wrong checkpoint donor;
- unrelated prompt donor;
- wrong layer;
- dose match;
- norm and manifold-distance diagnostics.

## 9.6 Interpretation

- Think clean patch rescues Think, Instruct patch does not: checkpoint-specific content or coding.
- Common-coordinate Think component rescues Instruct: representation exists and Instruct can consume it under forced supply.
- Think receiver output rescues Instruct while source component does not: receiver transformation is key.
- Neither cross-patch works but within-model rescue does: co-adapted geometry or off-manifold issue; do not infer absence.

---

# 10. O8: phase-resolved generation policy

## 10.1 Why this matters

The teacher-forced LP assay is the clean causal core, but Think post-training changes a generation policy. The side study should determine whether J dependence lives in prompt processing, explicit reasoning, or final serialization.

## 10.2 Models and templates

- 3.1 Think with its official reasoning template;
- 3.1 Instruct with its official chat template;
- optional 3.0 Think if parser/template behavior is stable;
- Base only for teacher-forced or plain generation unless a justified template exists.

## 10.3 Phases

For Think:

```text
prefill
reasoning
final answer
```

For Instruct:

```text
prefill
final answer
```

Do not impute a structurally absent phase.

## 10.4 Interventions

- prefill-only span-safe J;
- reasoning-only span-safe J;
- final-answer-only span-safe J;
- exact phase-matched rank/energy controls;
- optional selected-component patch rescue.

Hooks must be stateful, token-indexed, and tested against exact delimiter goldens.

## 10.5 Endpoints

- final accepted-answer accuracy;
- full answer-sequence LP;
- reasoning closure and parse status;
- generated token count;
- answer commitment time;
- bridge token appearance time;
- CoT lead with foil controls;
- clean-versus-ablated reasoning similarity;
- recovery after externally supplied bridge/rationale.

## 10.6 Filler and rationale controls

To distinguish semantic externalization from extra compute tokens:

- correct concise rationale;
- wrong rationale;
- shuffled rationale;
- length-matched filler;
- bridge-only supplied state;
- no rationale.

Use exact token matching where possible.

---

# 11. O9: intermediate training stages or controlled continuation

## 11.1 Inventory before experimentation

Search official OLMo release artifacts and manifests for checkpoints that satisfy:

- same 32B base lineage;
- exact training stage documented;
- compatible architecture and tokenizer;
- weights available;
- no silent change of base pretraining checkpoint;
- enough provenance to place the checkpoint on the graph.

Do not substitute a differently sized model or unrelated release and call it an intermediate.

## 11.2 If genuine intermediate checkpoints exist

Run a reduced frozen assay:

- transport gate;
- G5 on known development banks;
- Base/common-lens and own-lens geometry;
- Bank-S direct/composed core grid;
- corrected capacity at layers 24/32/40;
- Bank-W development primary cells if capability permits.

Use one common corpus and exact controls.

## 11.3 If no genuine intermediates exist

Do not fabricate a causal lineage. Two lawful alternatives exist:

### Controlled adapter continuation

Train small, reproducible adapters from Base under clearly distinct objectives:

- generic instruction SFT;
- explicit reasoning SFT;
- preference optimization without reasoning traces;
- matched-token-budget control.

This asks whether an accessible controlled intervention can move the same geometry/utilization axes. It does not reproduce the production OLMo recipe.

### Checkpoint-dense small-model analogue

Use an OLMo model size with released or trainable checkpoint density to study the temporal emergence mechanism, then treat 32B as the external validity target.

## 11.4 Training-study safeguards

- training data and seeds frozen before evaluations;
- evaluation families excluded from training;
- equal token budgets where comparing objectives;
- capability and perplexity tracked;
- multiple seeds where feasible;
- no selecting the best checkpoint on the J-space endpoint;
- report trajectory over training steps, not only final endpoints;
- separate adapter geometry from base-weight geometry.

This is likely a later phase. The first OLMo side-study block should inventory, not begin a large training run.

---

# 12. Per-checkpoint transport validity

## 12.1 Import the exact-JVP harness design

Use the same control-calibrated exact JVP versus secant framework planned for Gemma, but on OLMo as a positive/control family.

For each checkpoint, use:

- representative layers 24/32/40 plus late control;
- random, activation-aligned, J-selected, answer, and bridge directions;
- single-position perturbations matching the causal intervention;
- optional uniform-position perturbations matching the fitted-J estimand;
- fp32-delivered epsilon ladder;
- final residual and pre-softcap logit targets;
- homogeneity, odd symmetry, additivity, tangent error, and gain.

## 12.2 Why single-position and uniform probes both matter

The existing campaign found that a uniform shift can look nearly linear because it moves all keys together and partially cancels inside attention, while a position-wise intervention is less linear at shallow depth. The lineage comparison must use the probe that matches the actual ablation.

Report both, but causal interpretation uses the single-position result.

## 12.3 Decision

If transport fidelity differs materially across checkpoints and predicts lesion strength, include it as a moderator. If all checkpoints pass similarly, it becomes a strong control against instrument-driven trajectory.

---

# 13. Statistics and inference

## 13.1 Units

- canonical family is the default inferential unit;
- fact-level pairing for direct/composed and cross-checkpoint common support;
- prompt-level unit for generic activation/lens-fit corpora;
- training seed for any controlled continuation.

## 13.2 Intervals

- family-resampling percentile intervals for equal-family effects;
- exact sign flips when family count permits;
- plus-one Monte Carlo p-values otherwise;
- randomization-compatible intervals when making decision claims;
- TOST only with a precommitted SESOI and sufficient power;
- no normal interval labeled bootstrap.

## 13.3 Multiple testing

This side study is development and mechanism estimation. Define a small set of named hypotheses and control secondary discovery with BH-FDR within coherent families. Do not create one giant FDR pool spanning geometry, capacity, Bank W, receiver search, and training.

Receiver discovery must use discovery/validation family splits.

## 13.4 Hierarchical models

Use hierarchical models as synthesis tools, not as replacements for paired randomization:

- random intercept for family;
- optional fact nesting;
- fixed effects for checkpoint, task cell, load, derivation, redundancy, and frame;
- prespecified interactions;
- robust or Student-t residual sensitivity for heavy tails.

## 13.5 Missingness

- capability failure is a baseline-defined population property;
- no intervention outcome imputation for ineligible items;
- report source population, eligible population, and common-support population;
- never treat missing intervention rows as zero effect;
- no post-outcome dropping of difficult families.

---

# 14. Reproduction and evidence discipline

## 14.1 Per-row schema

Every model-backed row should contain:

```text
study_id, evidence_id, tier
model_id, revision, tokenizer hash
lens id/hash, lens frame, readout frame
bank version/hash, partition, family, fact, item, seed
prompt/token hashes, accepted aliases
condition, layer, position/phase
requested/achieved rank, removed energy, protected overlap
baseline LP, arm LP, candidate scores, accuracy
capability and commitment variables
state/checkpoint version
code commit, config hash, environment hash
```

## 14.2 Independent reconstruction

For every headline table:

- reconstruct from raw rows in an isolated output root;
- compare every estimate and interval;
- hash bootstrap distributions where deterministic;
- regenerate figures;
- run one model-cell sentinel from a clean process;
- verify imported source hashes.

## 14.3 Side-track release bundle

Final bundle:

```text
release/OLMO_LINEAGE_STATE_OF_RECORD.md
release/IMPORT_BUNDLE_PHASE4.json
release/IMPORT_BUNDLE_PHASE4.md
release/olmo_lineage_live_inventory.json
release/olmo_lineage_environment_lock.json
```

The import bundle must distinguish:

- Phase 4-needed capability evidence;
- broader development mechanism evidence;
- methods-only transport/geometry evidence;
- outputs forbidden from confirmatory use.

---

# 15. First 24-hour execution plan

## CPU lane, start immediately

1. Create side package, registry, run root, and foundation manifest.
2. Import and verify the four model/lens/grid manifests.
3. Audit existing lens corpora and recipes.
4. Freeze Base-capacity config and margins.
5. Freeze OLMo Bank-W capability configs.
6. Prepare Bank-W intervention configs on development families.
7. Implement Phase 4 import-bundle writer.
8. Prepare same-corpus lens corpus and decide which refits are actually missing.
9. Add tests before the first GPU result.

## GPU lane, priority order

### O1

Run 3.1 Think Bank-W capability, register, commit, push.

Run 3.1 Instruct Bank-W capability, register, commit, push.

Compute joint support with Qwen and release the early Phase 4 import bundle.

### O2

Run corrected Base capacity at layers 24/32/40 under the frozen activation corpus.

Run or reconstruct the same capacity table for 3.0 Think, 3.1 Think, and 3.1 Instruct under identical code and inputs.

### O4

Run the three-arm Bank-W development grid on Base and 3.0 Think first, then 3.1 Think and Instruct if time. Use resumable per-model state and register each model separately.

### O3

If the lens audit shows exact corpus mismatch, begin the highest-information same-corpus refit:

1. Base n=120 if missing;
2. 3.1 Think n=120 if missing;
3. Instruct and 3.0 Think next.

If existing lenses are exact same-corpus, skip refitting and run geometry analyses instead.

## Never-drop items

- isolated foundation;
- two Phase 4 Bank-W capability gates;
- early import bundle;
- Base capacity;
- lens provenance audit;
- at least one complete Bank-W development model cell.

## Drop order under pressure

```text
controlled continuation inventory beyond metadata
-> cross-checkpoint patching
-> full receiver search
-> phase-resolved generation
-> n=250 lens fits
```

Do not drop Base capacity or capability gates to begin a glamorous receiver search.

---

# 16. Coding-agent task queue

## Queue O0

- [ ] scaffold package and registry;
- [ ] implement immutable import resolver;
- [ ] enforce side run root;
- [ ] prohibit Phase 4 registry writes;
- [ ] foundation event and tests.

## Queue O1

- [ ] port Bank W candidate scoring exactly;
- [ ] Think capability config/run;
- [ ] Instruct capability config/run;
- [ ] joint-support analysis;
- [ ] Phase 4 import bundle;
- [ ] independent reconstruction.

## Queue O2

- [ ] activation corpus manifest;
- [ ] Base capacity config;
- [ ] four-checkpoint symmetric estimator;
- [ ] random-dictionary seeds;
- [ ] capacity table/figure;
- [ ] own/common-frame sensitivity.

## Queue O3

- [ ] lens provenance inventory;
- [ ] same-corpus classification;
- [ ] minimal refit plan;
- [ ] operator/token/selection geometry producer;
- [ ] selection-margin audit;
- [ ] geometry figures.

## Queue O4

- [ ] freeze Bank W contrast coding;
- [ ] three-arm resumable grid;
- [ ] model-by-load/derivation/redundancy analysis;
- [ ] capability-adjusted sensitivities;
- [ ] branch interpretation.

## Queue O5/O6

- [ ] crossed dictionary producer;
- [ ] continuous commitment variables;
- [ ] common-support matched-difficulty analysis;
- [ ] factor-specific contrasts.

## Queue O7/O8

- [ ] within-model rescue positive control;
- [ ] receiver screen/validation split;
- [ ] cross-checkpoint alignment and patch controls;
- [ ] phase parser and hook goldens;
- [ ] generation endpoints.

## Queue O9/O10

- [ ] official intermediate checkpoint inventory;
- [ ] no-substitution decision record;
- [ ] state-of-record report;
- [ ] independent reproduction;
- [ ] final import bundle and claim ledger.

---

# 17. Paper-facing claim ladder

## Level 1, already licensed at development tier

> In the tested OLMo 32B lineage, span-safe Bank-S J dependence is absent at Base, appears by the first released Think endpoint, persists at 3.1 Think, and is absent at the 3.1 Instruct sibling; the pattern survives a frozen base-lens frame and exact common-support fact pairing.

## Level 2, licensed if O2 and O3 support it

> The trajectory reflects changed causal recruitment of a broadly conserved thin J-readable coordinate system rather than a large increase in measured sparse capacity or a checkpoint-specific lens artifact.

Requires:

- Base capacity table;
- same-corpus geometry;
- per-checkpoint transport validity;
- no material capacity growth that contradicts the sentence.

## Level 3, licensed if Bank W supports it

> Think-path post-training is associated with external-state substitution: internally derived high-load state relies more on the J-readable channel, while supplied or redundant prompt state reduces that reliance.

Requires:

- capability gates;
- controlled Bank W development pattern;
- later untouched Phase 4 confirmation for a binary claim.

## Level 4, licensed if receiver work supports it

> The main post-training change lies in downstream consumption of a conserved representation, localized to specific receiver components and demonstrated by lesion rescue.

Requires:

- within-model causal rescue;
- receiver necessity;
- validation families;
- cross-frame controls.

## Prohibited without controlled training

- “Think training creates a global workspace.”
- “The reasoning objective is the causal variable.”
- “Instruct lacks a verbalizable channel.”
- “Capacity is unchanged from pretraining” before Base capacity is measured.
- “The dictionaries are identical” based only on common-frame effect stability.
- “A receiver is the workspace” based on one patch.

---

# 18. Completion criteria

This side study reaches its first release boundary when:

- [ ] isolated package, run root, and registry are verified;
- [ ] Phase 4 Bank-W capability gates for both OLMo 3.1 checkpoints are complete;
- [ ] all-model joint support is calculated and imported;
- [ ] Base capacity is measured with the corrected estimator;
- [ ] symmetric four-checkpoint capacity table is produced or clearly marked incomplete;
- [ ] every existing lens has a provenance/comparability classification;
- [ ] same-corpus geometry is measured with no unnecessary refit;
- [ ] at least the core Bank-W development grid is complete across the four checkpoints;
- [ ] continuous accessibility analysis is complete on common support;
- [ ] transport validity is checked per checkpoint or explicitly queued;
- [ ] receiver work has either a validated positive result or an honest bounded null;
- [ ] intermediate-checkpoint availability is documented without substitution;
- [ ] figures regenerate from registered tables;
- [ ] independent reconstruction passes;
- [ ] `OLMO_LINEAGE_STATE_OF_RECORD.md` and the import bundle are complete.

The side study should stop after this boundary and hand its results to the Phase 5 router. It should not grow into an unbounded fifth model matrix merely because the OLMo questions produced interesting new branches.
