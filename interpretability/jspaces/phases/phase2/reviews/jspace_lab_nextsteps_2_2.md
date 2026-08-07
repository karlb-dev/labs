# jspace_lab_nextsteps_2_2.md

## Forensic branch review and the next confirmatory execution plan

**Review target:** `karlb-dev/labs`, branch `interp_jspace_part2` at the end-of-VM7 checkpoint `d4f2c69`, compared with `main` at `4097c44713d0084d4da3d3c084e79aed2068c740`  
**Branch delta reviewed:** 104 commits ahead of `main`, with no commits behind  
**Context reviewed:** the Part 2 handout, `REPORT_PART2.md`, `SCIENTIFIC_PREREGISTRATION_DRAFT.md`, `inprogress.md`, the previous forensic addendum, the repair preregistration, the new `jspace_part2` package, its tests, the evidence registry, and the key pilot result files  
**Purpose:** give the next coding/research agent an exact, publication-oriented continuation plan that preserves the valuable pilot results while repairing the remaining blockers before any confirmatory data are exposed

> **Paste-line for the coding/research agent**  
> Read this file after the previous Part 1 and Part 2 plans, the original forensic addendum, `REPAIR_PREREGISTRATION.md`, the current scientific preregistration draft, `REPORT_PART2.md`, and `inprogress.md`. Treat every currently banked Part 2 result as pilot, exploratory, dev, or methods evidence exactly as registered. Do not freeze the scientific preregistration and do not run a confirmatory intervention cell yet. First complete the reopened correctness gates in Sections 3 through 7 below, especially the family-cluster repair, the R2 centered-variance correction, the protected-generation alignment fix, the real fresh-VM reproduction path, the per-context-J study, and G5. Preserve all old artifacts and supersede them by new evidence IDs rather than editing history. The paper-faithful averaged-J assay remains the primary replication arm even if a better context-specific estimator is found; any improved estimator becomes a separately named methods arm. Stop at the final preregistration candidate and request principal-investigator approval before the dedicated freeze commit.

---

# 0. Executive verdict

The branch is a substantial scientific and engineering improvement over `main`. It did not merely run more models. It converted an exploratory notebook-like campaign into the beginnings of a real research package, added output protection, rank-safe projectors, full-sequence scoring, paired analysis, an evidence registry, a hard-item bank, a positive control, cross-model pilots, lens-stability checks, and explicit preregistration governance. It also caught and withdrew several faulty analyses instead of quietly burying them. That error-correction behavior is one of the strongest parts of the project.

The pilot campaign has produced several genuinely interesting results:

1. Protecting the clean output top-10 removes most of the catastrophic live-ablation effect, so the bulk of the Part 1 collapse was output-stream deletion rather than a selective workspace lesion.
2. A smaller protected J-specific tail remains on factual items, including cases where the answer token itself was protected.
3. Qwen shows a much cleaner two-hop-versus-one-hop shape than the tested OLMo checkpoints under the repaired pilot protocol.
4. The causal shape varies along the OLMo lineage while the current occupancy estimate is comparatively flat.
5. The plain logit dictionary reproduces a substantial fraction of the frozen-J effect, narrowing the causal role attributable specifically to the Jacobian transport.
6. Gemma is a valuable methods boundary case: its mid-band output basis is opaque, the fitted J-lens does not rescue that opacity, and the tested finite-scale transport behaves differently from OLMo.
7. The branch has enough hard one-hop material to build a strong confirmatory pool after the partition problem is fixed.

However, the campaign is **not yet ready to freeze the scientific preregistration**. The review found several hard blockers that affect the reported pilot intervals, the capacity headline, the claimed reproduction contract, and the generation implementation. The most serious are:

- the two-hop `family` field used by every paired clustered analysis is not the true generation unit;
- R2 claims globally centered excess variance but computes raw-energy shares and never uses the supplied global mean;
- the protected free-generation path broadcasts the final prompt protection set across every prompt position;
- the current evidence registry supersession design can erase the metadata needed to reproduce an evidence item;
- the advertised fresh-VM one-command reproduction path does not actually fetch or pin the declared heavy inputs and cannot reproduce exact result hashes because volatile timestamps are embedded in the hashed files;
- the Stage-3 bank clears the item/family floor only before partitioning, so it cannot support two genuinely independent family-level confirmatory and replication sets;
- G5 has not run;
- the per-position versus averaged-J experiment that may change the methods interpretation has not run;
- the random dynamic control is a useful mechanics control but is not energy-, spectrum-, or geometry-matched, despite some report wording calling it matched;
- the local-linearity prose overstates what a finite-scale bf16 superposition test establishes.

The appropriate current status is therefore:

> **Repair era scientifically productive, but confirmatory freeze blocked by a small number of identifiable correctness repairs.**

That is not a retreat. The branch has successfully compressed a sprawling mystery into a sharp set of solvable questions. The next phase should be less of a model safari and more of a clean-room assay build.

---

# 1. What changed against `main`

## 1.1 The branch has two layers of new work

The branch introduces both:

1. an exploratory publication mirror under `interpretability/jspaces/phases/phase1/part2_exploratory/`, containing the living report, figures, handout, master tables, mirrored results, and operational plans; and
2. a confirmatory-capable package under `interpretability/jspaces/phases/phase2/`, containing reusable Python modules, configs, tests, a CLI, provenance machinery, an evidence registry, and the scientific preregistration draft.

The split is conceptually sensible: the mirror tells the research story, while the package should be the executable source of truth. In practice, the two trees have started to drift, and the next pass must make the package authoritative and the mirror generated-only.

## 1.2 Major improvements worth keeping

The following changes are strong and should remain foundational:

- `REPORT_v2_ERRATA.md` explicitly corrects Part 1 rather than laundering old claims into the new study.
- `REPAIR_PREREGISTRATION.md` separates assay repair from scientific confirmation.
- `SCIENTIFIC_PREREGISTRATION_DRAFT.md` blocks confirmatory execution until explicit design decisions are made.
- `orthonormal_basis_from_rows` replaces raw QR with rank-revealing SVD.
- `RunningVectorMoments` supplies mergeable global covariance state.
- `conditional_sequence_logprob` moves the primary readout beyond first-token-only scoring.
- `ProtectedDynamicAblator` implements clean-output protection and exact span projection.
- selected-id logging makes the protected tail mechanistically inspectable.
- `effective_gain` correctly avoids assuming one RMSNorm parameter convention across model families.
- the evidence registry and provenance blocks create the right research-accounting abstraction, even though the implementation needs strengthening.
- the Stage-3 item bank is authored family-first and explicitly recognizes pseudo-replication risk.
- the package contains real synthetic and tiny-model tests rather than relying entirely on 32B runs.
- the project repeatedly records withdrawn analyses and the reason for withdrawal.

## 1.3 The most important pilot results, with current evidence status

| Result | Current status | What it supports now | What remains before confirmation |
|---|---|---|---|
| Output protection flips the median live effect toward zero | Strong pilot | Most Part 1 live damage involved deleting output-aligned directions | Correct generation path; exact paper-mechanics crosswalk; geometry-matched controls |
| Protected J tail remains on a minority of items | Strong pilot | A J-specific indirect-content effect is plausible | Correct family inference; independent item split; matched controls; endpoint freeze |
| Qwen shows forward dissociation shape | Strong pilot | The protected effect is model-dependent | Corrected clustered statistics; hard one-hop cohort; confirmatory Qwen cell |
| OLMo lineage changes causal shape | Exploratory-pilot | Post-training regime is a serious moderator | Actual primary 3.1 Think/Instruct pair; own lenses; no strict order claim until endpoint fixed |
| Logit dictionary reproduces about half the frozen-J effect | Strong exploratory control | Output-aligned geometry explains substantial effect | Add to every confirmatory model; compare against label-shuffled and J-rotated controls |
| OLMo occupancy near 2, Qwen 3-4 | Provisional methods result | Current sparse estimator sees low open-model occupancy | Fix centered excess share; solver and crossing audit; uncertainty intervals |
| Gemma mid-band opacity and failed J rescue | Strong methods pilot | This J-lens recipe does not transfer cleanly to Gemma | Narrow nonlinearity wording; no Gemma causal confirmation under this assay |
| Mean-J predicts only part of OLMo transport | Strong methods pilot | Averaging may discard context-specific transport information | Run exact per-context JVP/VJP study before freeze |

---

# 2. Findings that require immediate correction

## 2.1 P0 blocker: the clustered-family labels in the R7 pilot are wrong

`jspace_part2/battery.py` assigns the two-hop family as:

```python
"family": it["name"].split("-")[0]
```

This is not a stable definition of the data-generating unit. Examples from the released set make the problem visible:

- `animal-cover-turtle`, `animal-legs-buffalo2`, and `animal-nose-elephant` all become `animal`, although they use different relations and prompt templates;
- several unrelated items whose upstream `category` is `multihop` become singleton prefixes such as `amazon`, `atomic`, and `basketball`;
- other names happen to share a first word even when the relation differs.

Every current family-clustered interval, tail-rate interval, variance decomposition, ICC estimate, and G6 power simulation reads that field. Therefore, the scientific conclusion may survive, but the reported uncertainty and sample-size design cannot be treated as valid until the family map is repaired.

### Required repair

Create a versioned, human-audited mapping for every released probe-swap item:

```json
{
  "item_name": "animal-cover-turtle",
  "upstream_category": "animal-cover",
  "canonical_family": "animal_covering",
  "template_id": "animal_covering_v1",
  "family_rationale": "same relation and surface template as ..."
}
```

Do not blindly replace the current field with `category`. The upstream `category` is useful, but generic categories such as `multihop` may still combine heterogeneous relations. The mapping must identify the actual reusable relation/template unit.

Then:

1. preserve every old parquet and JSON;
2. create a corrected per-item view by joining the immutable score rows to the new family map;
3. register superseding evidence IDs;
4. recompute R7 paired intervals for all four models;
5. recompute the tail-rate endpoint;
6. recompute G6 power;
7. regenerate the ladder, handout, report, and preregistration design numbers;
8. add a test that fails if any item lacks a canonical family or if two supposedly distinct families share the same normalized template hash.

### Gate

No endpoint, sample-size, ICC, or confidence-interval decision may be frozen until this recomputation lands.

## 2.2 P0 blocker: R2 does not yet compute the quantity its prose claims

`occupancy_and_excess` computes `hc = h - global_mean`, but the centered activation is never used. Both `share_j` and `share_r` use raw `h` energy. The function comment itself acknowledges that globally centered variance is deferred, while the report and handout describe the result as the paper-defined, globally centered excess variance estimate.

This means two different quantities are currently fused under one label:

- occupancy from sparse reconstruction marginal gains;
- raw reconstruction-energy advantage at median occupancy.

The occupancy number may remain informative. The excess-variance number is not yet the preregistered estimand.

There are additional solver concerns:

- the pursuit chooses the maximum correlation even when every untaken correlation is non-positive;
- rows with zeroed coefficients can still consume nominal support slots;
- the synthetic validation covers mostly low-coherence mixtures;
- the noise test reuses random-control gains computed on a different synthetic activation set;
- only a single example is checked against NNLS on known support;
- the custom two-consecutive-crossing rule is a preregistered branch choice, not demonstrated to be the paper's exact rule;
- no bootstrap interval over prompts or random dictionaries is reported.

### Required repair

Implement `occupancy_v2.py` with three separately named outputs:

1. `occupancy_crossing_k`: sparse support crossing against random controls;
2. `raw_reconstruction_excess`: retained for continuity with the pilot;
3. `centered_variance_explained_excess`: the confirmatory capacity endpoint.

For the centered endpoint, freeze one explicit formula after a paper-method crosswalk. At minimum report both of these candidates on dev data before choosing:

```python
# Candidate A: centered reconstruction variance share
var_h = ((H - H.mean(0)) ** 2).sum()
var_r = ((R - R.mean(0)) ** 2).sum()
share = var_r / var_h

# Candidate B: centered R-squared
Hc = H - H.mean(0)
Rc = R - R.mean(0)
r2 = 1.0 - ((Hc - Rc) ** 2).sum() / (Hc ** 2).sum()
```

The paper-facing report must name the chosen definition, explain why it matches the paper's analysis, and preserve the other as sensitivity.

The pursuit must stop or skip support growth for a row when no positive residual correlation remains. Variable support sizes must be represented explicitly rather than allowing zero-gain atoms to burn `K`.

Add tests for:

- positive-support exhaustion;
- high-coherence and near-duplicate dictionaries;
- the same null activations with their own random controls;
- batch splitting and resume equivalence;
- fp32 versus bf16 selection agreement;
- multiple support sizes and signal-to-noise levels;
- comparison against a high-accuracy NNLS reference across at least 100 synthetic rows;
- centered-variance recovery on synthetic data with a known mean shift;
- crossing-rule sensitivity to persistence `1`, `2`, and `3`.

### Gate

The capacity headline, HP4 wording, and any comparison with Claude must remain provisional until corrected R2 artifacts supersede the current files.

## 2.3 P0 blocker: protected free generation misaligns the clean protection sets

The teacher-forced path builds a top-k protection set per sequence position. The free-generation path does not do that for the initial prefill. It takes only the final prompt logit vector and broadcasts that one protection set across every prompt position during the ablated prefill.

That means future chat-mode or generation-based confirmatory cells would not be running the intended protocol.

A second issue is that the selected dose is reduced globally using the minimum number of available positive rows across every flattened position. One unusual position can shrink `k` for the entire sequence. Selection must be row-wise.

The logging fields also need semantic repair:

- `n_steps` currently counts layer-hook applications, not decode tokens;
- `protected_hits_blocked` counts protected rows with finite positive scores, not protected rows that actually would have entered the top-k;
- `last_selected` preserves only the first flattened position;
- mean removed energy pools layers, positions, and forwards into one ambiguous number.

### Required repair

Implement a new versioned path rather than silently changing the existing pilot implementation:

```python
clean = hf(input_ids=prompt_ids, use_cache=True)
prefill_protect = clean.logits[0].topk(protect_k, dim=-1).indices  # [T, pk]

# Either:
# A. paper-all-positions prefill, using the full [T, pk] matrix; or
# B. decode-only intervention, with hooks explicitly inactive during prefill.
```

Selection must keep a row-specific validity mask. Do not use `avail.min()` as a global dose. The implementation should return per-position requested `k`, available positive count, selected count, effective rank, singular spectrum, and removed energy.

Add a real-transformer golden test that proves:

1. per-position protection sets align with the logits from that same position;
2. decode-only mode never alters the prompt KV cache;
3. prefill-only mode never fires on decode;
4. `both` equals the explicit composition of the two phases;
5. one low-availability position does not shrink other positions;
6. the protected token IDs never appear in selected IDs at their positions;
7. capture on versus off is bit-identical;
8. interrupted generation reproduces uninterrupted generation under deterministic decoding.

### Gate

No generation-based confirmatory endpoint may run until this path passes.

## 2.4 P0 blocker: the advertised reproduction contract is not yet executable

The intended contract is excellent. The implementation currently verifies local files, not reproducibility from a clean machine.

Specific gaps:

- `pyproject.toml` uses broad lower bounds rather than a lock and omits dependencies used by the package, including SciPy and the pinned `jlens` source.
- configs and registry rows contain absolute `/content/...` paths.
- `repro.sh` installs the current checkout, runs tests, and verifies current local outputs. It does not check out the evidence item's code commit in an isolated worktree.
- it does not fetch, materialize, or refit declared heavy inputs;
- it does not prove that the loaded local model directory corresponds to a pinned Hugging Face revision;
- `resolve_model` records local directories with `revision: null`;
- cached Hub resolution chooses the lexicographically latest snapshot, not necessarily the snapshot loaded by the producer;
- `Provenance.allow_dirty` is stored but not enforced centrally by `Provenance.block`;
- `created_utc` is embedded in the output file whose exact SHA is registered, so an exact rerun cannot reproduce the same file hash;
- the golden test claims hash stability but writes only once and does not compare two canonical payloads;
- `inventory.py` skips paths that were previously seen without checking whether size or modification time changed;
- files over the default size threshold are left without a content hash;
- registry supersession appends a partial row under the same `evidence_id`; because lookup is last-row-wins, a supersession event can hide the original command and outputs;
- `registry-list` assumes `tier` is always a string, which is false for the partial supersession rows;
- the repro command reruns a producer but does not verify the newly generated outputs afterward.

### Required repair

Use an event-sourced registry with distinct event types:

```json
{"event":"evidence_created", "evidence_id":"...", "schema_version":2, ...}
{"event":"evidence_superseded", "evidence_id":"old", "superseded_by":"new", ...}
{"event":"evidence_reproduced", "evidence_id":"...", "runner":"...", ...}
```

`registry_find` must reconstruct the evidence object from its creation event and attach later status events without replacing its creation metadata.

Separate deterministic scientific payload from volatile provenance:

```json
{
  "payload": {... deterministic science ...},
  "payload_sha256": "...",
  "provenance": {... timestamps, machine, command ...}
}
```

Verify `payload_sha256` exactly. Verify the complete envelope only by schema and declared tolerance.

Create:

- `constraints.txt` or a lock file with exact versions and hashes;
- a pinned `jlens` install from commit `581d3986...`;
- logical artifact URIs such as `drive://special-lab-1/...` instead of hard-coded local paths;
- a resolver that maps logical URIs to local mounts, release assets, or a documented refit recipe;
- model manifests containing Hub ID, exact revision, config hash, tokenizer hash, and either safetensor index hash or per-shard hashes;
- an isolated-worktree repro path that checks out the recorded code commit;
- post-run payload verification;
- chunked hashes or a Merkle manifest for large lenses rather than `sha256: null`;
- schema validation for every registry append;
- a lock around registry writes;
- CI tests for registry creation, supersession, lookup, and reproduction.

Remove committed `jspace_part2.egg-info/` and add it to `.gitignore`.

### Gate

G0 is not fully passed until one evidence item can be recreated from a genuinely fresh environment using only the repository, the declared artifact resolver, and the registered command.

## 2.5 P0 blocker: the Stage-3 bank cannot yet support an independent split

The bank currently contains 126 hard items across 34 canonical families. That clears a single `n >= 90`, `families >= 30` floor. It does not support both:

- a confirmatory set with at least 30 independent families; and
- an independent replication set with at least 30 disjoint families.

Splitting items within the same relation templates gives an item holdout, not an independent family-level replication. Splitting families leaves too few families per side.

### Required repair

Choose one of these honest designs:

**Preferred:** expand to at least 60 canonical families, preferably 70, with at least three capable items per family. Split families, not items, into confirmatory and replication cohorts.

**Compute-saving fallback:** use 30-40 families for the confirmatory study and declare the second run an independent computational rerun on the same frozen sample, not a new-sample replication. Add a later external-family validation set as a separate study.

The preferred route is scientifically stronger and the GPU cost is modest because the protected teacher-forced grid is cheap relative to model loading.

Also define three capability cohorts before intervention outcomes are visible:

1. `lineage_anchor_cohort`: capable on the OLMo primary pair;
2. `cross_model_intersection_cohort`: capable on every model in the cross-model primary contrast;
3. `model_specific_cohort`: capable on one model, used only for within-model estimates.

This avoids selecting every cross-model item based solely on OLMo Think difficulty and then interpreting model differences as if the sample were neutral.

## 2.6 P0 blocker: G5 remains open

The task gate is not optional at publication quality. Run it rather than waiving it.

G5 must establish, before intervention outcomes:

- verified ground truth and source metadata for factual items;
- unambiguous accepted aliases;
- answer tokenization under every model;
- baseline full-sequence likelihood and generation accuracy;
- no answer string or trivial alias leakage in the prompt;
- no direct surface shortcut that makes a nominal two-hop item one-hop;
- successful bridge-swap or counterfactual test for two-hop items;
- matched prompt length and answer-token length summaries;
- canonical family and template hashes;
- capability masks and exclusion reasons;
- synthetic working-memory tasks with exact deterministic answers;
- paper-released task adaptation audits.

The gate output should be an immutable item manifest, not a report paragraph.

---

# 3. Important interpretation corrections

## 3.1 The dynamic random arm is not a matched control

The protected random dictionary is useful because it applies the same selection and projection machinery to a nonsemantic dictionary. It is not matched on the quantities that drive intervention severity. The pilot reports removed-energy differences on the order of `0.12%` for J versus `2.6%` for random in some cells. That is evidence that the J effect is not merely caused by removing more energy than random. It does not make the random arm a fair geometry-specific counterfactual.

Use precise names:

- `dynR_mechanics_control`, not `matched_random`;
- `dynJ_label_shuffled` for token-label semantics;
- `dynJ_rotated` for same-span or same-spectrum geometry;
- `dyn_spectrum_matched_nonJ` for a true geometry control;
- `dyn_energy_rank_matched_random` for dose matching.

The paper itself reports matched-norm control families in later ablations. A publication-quality open-model study should not rely on a single isotropic control.

## 3.2 The current protected ablator is paper-inspired but not yet proven paper-identical

The branch deliberately replaces the Part 1 two-pass deflation with exact SVD projection of the selected span. That is a principled repair, but it may not be mechanically identical to the paper's intervention. The report should not use “exact paper protocol” until a method crosswalk establishes:

- the paper's activation score;
- whether negative activations are excluded;
- whether selected rows are individually damped, sequentially projected, or jointly span-projected;
- the layer and position ranges;
- prefill versus decode behavior;
- clean-output protection alignment;
- intervention magnitude or norm matching;
- target-layer and normalization conventions.

Run both a `paper_literal` implementation and the `rank_safe_span` implementation on the same dev items if the paper implementation can be reconstructed. If they agree, the safer span variant can be the confirmatory implementation with an equivalence claim. If they differ, the literal replication must remain primary and the rank-safe variant becomes a methods robustness arm.

## 3.3 Teacher-forced full-sequence scoring and free generation answer different questions

Teacher-forced answer-sequence likelihood is a valuable deterministic endpoint. It conditions each answer token on the correct preceding answer tokens. During a dynamic intervention, it also lets later answer positions be selected and protected in a context that already contains the gold answer prefix.

That is not the same estimand as whether the model freely produces the correct final answer under intervention.

Recommended endpoint roles:

- raw-completion primary measurement: deterministic full-sequence conditional likelihood, clearly named as such;
- raw-completion behavioral secondary: exact/alias generation accuracy under deterministic decode;
- chat-mode primary: generation endpoint only, with a frozen answer extractor and budget;
- sampled robustness: fixed seeds and temperature, secondary only.

Keep the alias set frozen. Record the variant selected by any `max` aggregator. For the confirmatory primary, prefer either:

1. a canonical answer string frozen before outcomes; or
2. `logsumexp` over a frozen accepted alias set, which measures total probability assigned to the answer concept without allowing each arm to pick a different winning surface form.

Retain max-over-aliases as a sensitivity analysis if continuity with the pilot matters.

## 3.4 The local-linearity conclusion is too broad

The finite-scale superposition tests are useful. They show that at perturbation scales delivered faithfully in bf16, OLMo behaves approximately linearly in the tested aggregate residual-to-residual map, while Gemma shows substantial non-additivity under the same procedure.

They do **not** establish that “no Jacobian can model Gemma however well estimated.” A Jacobian is a local derivative. The test cannot resolve the arbitrarily small neighborhood because bf16 imposes a measurement floor. The supported statement is:

> At the tested prompts, layers, directions, and intervention-relevant finite scales, a first-order linear approximation of the chosen aggregate source-to-target residual map is substantially less accurate on Gemma than on OLMo.

The methods finding remains interesting. Narrower wording makes it stronger, not weaker.

The OLMo result also does not yet prove that all remaining error is caused by position/prompt averaging. Other contributors include finite-scale curvature, target definition, future-position aggregation, normalization, and estimator variance. That is why D2 must run.

## 3.5 Readout validity and causal transport validity must remain separate

A lens can rank a token well while predicting intervention response poorly, and vice versa. The branch itself observes this on Gemma. Future reports must maintain separate ledgers for:

- readout validity;
- dictionary identification across fit corpora;
- local transport faithfulness;
- causal intervention specificity;
- behavioral outcome.

No one rung should automatically upgrade the others.

---

# 4. Recommended decisions for D1 through D6

These are the recommended scientific choices. The agent should implement the supporting analysis and prepare the freeze candidate, but the dedicated binding commit still requires principal-investigator approval.

## D1: use a split endpoint strategy

Choose the spirit of option (a), but do not force one endpoint to answer every hypothesis.

### Binary primary for tail-carried hypotheses

For HP2 and HP3, use the paired J-versus-control tail-rate difference at a frozen one-nat threshold. This matches the observed mixture structure and is powerable.

### Continuous estimation for model ordering

For HP1, do not declare a strict base < Think < Instruct < Qwen order as a binary primary. The mean and rate endpoints already disagree about Think versus Instruct. Instead use:

- an omnibus model-by-task interaction;
- prespecified OLMo 3.1 Think versus OLMo 3.1 Instruct contrast;
- Qwen as an external validation contrast;
- per-model continuous deltas with confidence intervals;
- tail-rate estimates as a parallel descriptive view.

The pilot ladder remains the origin of the hypothesis, not a sequence that confirmation must reproduce exactly.

### Two-part secondary model

Because the distribution is a zero mode plus a heavy tail, add a prespecified hurdle-style secondary analysis:

1. probability of crossing the one-nat damage threshold;
2. magnitude of damage conditional on crossing it.

This describes the phenomenon more faithfully than a single mean without multiplying the primary family.

## D2: run the per-context-J study before freeze, but retain mean-J as the replication primary

The average J is the paper's object. Replacing it in the primary analysis would answer a different question and make the clean replication claim slippery.

Therefore:

- `meanJ_paper` remains the primary replication arm;
- `contextJ_methods` becomes a prespecified methods arm if it passes a dev gate;
- no confirmatory item may be used to decide whether context J passes.

The context-J study design is in Section 7.

## D3: keep Gemma as a methods boundary case

Do not run confirmatory J-direction causal cells on Gemma under the current instrument. Keep:

- fit identification;
- J-versus-logit readout by depth;
- finite-scale transport diagnostics;
- adapter and normalization lessons.

Do not spend the main confirmatory budget inventing a nonlinear Gemma lens. That would become a separate methods project.

## D4: freeze the one-nat threshold, with sensitivity visible

Retain `delta < -1.0 nat` as the binary tail threshold because it was used in the pilot and the sensitivity curve is already available. Freeze it before confirmatory outcomes. Report `0.5`, `1.5`, and `2.0` nat thresholds as prespecified sensitivity only, with no threshold shopping.

## D5: expand and partition at the family level

Expand to at least 60 canonical families. Use a deterministic, stratified family-level split. The split artifact must include:

- seed;
- source-bank hash;
- family list per partition;
- item list per partition;
- difficulty-stratum counts;
- answer-token-length counts;
- model capability-mask counts;
- SHA-256 of the final manifests.

No intervention result may be viewed between partition generation and the preregistration freeze commit.

## D6: run G5

Do not waive it. The task gate is cheaper than the interpretive damage caused by one shortcut-heavy family.

---

# 5. Revised confirmatory hypotheses

The current draft is close, but the hypotheses should be tightened after the corrected pilot statistics.

## HP1: post-training changes the protected-ablation task contrast

**Recommended wording:**

> Under the paper-faithful averaged-J output-protected dynamic ablation, the contrast between two-hop and difficulty-matched one-hop effects differs across the primary OLMo 3.1 Think and Instruct checkpoints. Qwen3.6-27B provides an external validation estimate. The primary claim is a model-by-task interaction and the prespecified Think-versus-Instruct contrast, not a strict four-model rank order.

**Primary endpoint:** continuous answer-sequence delta interaction.  
**Primary model pair:** `allenai/Olmo-3.1-32B-Think` versus `allenai/Olmo-3.1-32B-Instruct`.  
**Secondary:** tail-rate interaction and older lineage checkpoints.

## HP2: factual accessibility, not nominal hop count, predicts one-hop damage

**Recommended wording:**

> Within each model, protected-ablation damage on one-hop factual recall varies with preregistered baseline accessibility strata. The easy-versus-hard interaction is estimated independently of the two-hop comparison.

Use baseline full-sequence likelihood as the primary accessibility measure. Corpus frequency may be included only if it comes from a pinned, model-relevant corpus estimator and is not treated as causal.

## HP3: a protected internal-content tail exists on think-trained OLMo

**Recommended wording:**

> On the untouched think-model confirmatory set, the rate of items losing more than one nat under protected averaged-J ablation exceeds the corresponding rate under the primary geometry-matched control. The effect must also reproduce under an independently fitted lens on the replication partition.

The primary control must be defined before freeze. Isotropic random alone is insufficient.

## HP4: open-model occupancy lies below the reported Claude range under the corrected estimator

**Recommended wording:**

> Under the finalized sparse solver and centered excess-variance definition, the tested open models have median occupancy and excess explained variance below preregistered boundary values derived from the reported Claude range, with uncertainty over prompts and random dictionaries.

Do not call this a causal explanation of HP1 unless a formal moderator analysis supports it.

## HP5: high working-set load reveals or closes a static/persistent-span effect

Keep this as a separate confirmatory sub-study or a second multiplicity family. Do not let it inflate the primary R7 model-matrix family.

**Recommended wording:**

> On synthetic and released high-load tasks that pass G5, the ablation-by-load interaction is either directionally positive beyond the SESOI or statistically equivalent to zero within the frozen bound.

## HM1: context-specific transport improves causal-direction faithfulness

Add a methods hypothesis outside the five behavioral primaries:

> On held-out dev prompts, context-specific Jacobian actions predict measured source-to-target responses more accurately than the corpus-averaged J in the paper band, without using confirmatory items or outcomes.

This decides whether a context-J methods arm is included. It does not replace HP1's paper-replication arm.

---

# 6. Exact primary analysis table to freeze

The preregistration should contain a table like this, with no ambiguity left to the analysis agent.

| ID | Hypothesis | Population | Conditions | Endpoint | Estimand/test | Multiplicity |
|---|---|---|---|---|---|---|
| P1 | HP1 OLMo post-training interaction | primary pair, cross-model-intersection task cohorts | baseline, protected mean-J, primary matched control | full-seq delta | mixed model `delta ~ model * task + (1|family)` plus frozen Think-Instruct interaction contrast | Holm family A |
| P2 | HP2 accessibility | hard/easy one-hop strata within primary pair | same | full-seq delta | model-specific accessibility-by-ablation interaction | Holm family A |
| P3 | HP3 protected tail | OLMo 3.1 Think confirmatory partition | protected mean-J vs primary matched control | `I(delta < -1)` | paired family-level tail-rate difference | Holm family A |
| P4 | HP4 capacity boundary | held-out descriptive corpus | J versus random dictionaries | occupancy and centered excess share | bootstrap CI and one-sided boundary test/equivalence as frozen | Holm family A or separate methods family |
| P5 | HP5 load engagement | C1 high-load set | static/persistent J vs matched controls | accuracy/lp interaction | load-by-ablation interaction; equivalence if null | separate family B |

Specify the mixed-model optimizer, random-effect structure, convergence fallback, and whether families receive equal or item-proportional weight. The current cluster bootstrap implicitly weights larger families more heavily. Choose the estimand explicitly.

For null claims, use a proper TOST or its confidence-interval equivalent. At alpha `0.05`, equivalence corresponds to a `90%` two-sided interval inside the SESOI, not an ordinary `95%` interval.

---

# 7. Execution plan

## Stage N0: freeze the current pilot state before changing code

**Compute:** CPU only.  
**Goal:** make the present branch a fully recoverable historical checkpoint.

1. Tag the current head, for example `jspace-part2-pilot-vm7`.
2. Export the current evidence registry to a canonical snapshot.
3. Hash all small mirrored results and figures.
4. Create chunked manifests for every large lens and raw parquet used by a claim.
5. Record the exact current report and handout hashes.
6. Mark the current scientific preregistration draft as `DRAFT_V2_PRE_REVIEW`, without freezing it.
7. Add a short `NEXTSTEPS_2_2_ACCEPTED.md` that points to this file once the user approves the continuation plan.

**Do not recompute or overwrite any pilot artifact in this stage.**

## Stage N1: repair the package and reopen the affected gates

**Compute:** CPU, plus tiny-model GPU test.  
**Order:** do this before D2 so new methods evidence is born under a real repro contract.

### N1.1 Repository source of truth

- Make `interpretability/jspaces/phases/phase2/` authoritative for executable code.
- Reduce `interpretability/jspaces/phases/phase1/part2_exploratory/code/` to generated mirrors or pointers.
- Add a generated-file banner to mirror files.
- Remove `jspace_part2.egg-info`.
- Add package, cache, checkpoint, and local-weight exclusions to `.gitignore`.
- Generate report/handout status from one machine-readable campaign-state file.

### N1.2 Registry and provenance v2

Implement the event schema described above. Add JSON Schema or Pydantic validation. Migrate existing registry rows into a versioned snapshot without deleting the original JSONL.

Required tests:

- create evidence;
- append supersession;
- lookup still returns original command and outputs;
- registry listing handles partial events;
- duplicate creation is rejected;
- concurrent append is lock-safe;
- dirty-tree handling is enforced by the writer, not by runner convention;
- exact scientific payload hash is stable across two runs with different timestamps.

### N1.3 Real reproduction

Implement:

```bash
bash interpretability/jspaces/phases/phase2/repro.sh <evidence-id> \
  --workspace /tmp/repro-<id> \
  --fetch-heavy auto \
  --verify
```

The command must:

1. resolve the immutable evidence creation event;
2. create an isolated worktree at `code_commit`;
3. install exact locked dependencies;
4. resolve and verify every declared input;
5. verify model revision and tokenizer;
6. run the producer command;
7. verify deterministic payload hashes or preregistered numeric tolerances;
8. emit a reproduction event without mutating the original evidence.

Use one CPU evidence item and the SmolLM golden evidence as acceptance tests on a fresh VM/container.

### N1.4 Repair protected generation

Implement `protected_dynamic_v2.py` with per-position prefill protection, row-wise dose, phase control, and structured logs. Keep the pilot implementation importable for exact historical reproduction.

### N1.5 Repair family metadata

Add `data/probe_swap_family_map.json` and `family.py`. Every row producer must obtain family from a manifest, never derive it from string splitting.

### N1.6 Repair occupancy

Implement `occupancy_v2.py`, solver tests, centered excess share, and uncertainty.

**N1 gate:** rerun the package self-tests and the fresh-VM golden. G0, G1, and G3 remain reopened until all pass.

## Stage N2: recompute the pilot analyses under corrected metadata and R2

**Compute:** mostly CPU; occupancy reruns require model loads.  
**Tier:** pilot, explicitly superseding old evidence.

Run in this order:

1. corrected family map audit;
2. corrected R7 paired intervals for base, Think, Instruct, Qwen;
3. corrected tail-rate endpoint;
4. corrected G6 power simulation;
5. corrected ladder figure and handout;
6. corrected R2 on OLMo Think, OLMo Instruct, and Qwen;
7. corrected capacity errata figure;
8. update preregistration decision section with the new numbers.

Do not preserve conclusions by wording. If the corrected clusters widen intervals or reorder models again, report exactly that.

**N2 gate:** the endpoint recommendation and sample-size floor must be based on corrected evidence IDs only.

## Stage N3: run D2, the context-specific Jacobian study

**Compute:** approximately 1-3 GPU hours on OLMo, depending on directional batch size.  
**Tier:** methods pilot.  
**Data:** dev prompts only, with all confirmatory item hashes excluded.

### N3.1 Do not materialize a full Jacobian first

The first question can be answered with Jacobian-vector and vector-Jacobian products.

For each selected source layer and context:

- exact `J_context @ delta` via JVP for perturbation directions;
- exact token row `J_context.T @ u_token` via VJP for selected answer, bridge, foil, and random output directions;
- average-J prediction from the campaign lens;
- prompt-specific position-averaged J prediction if affordable;
- present-only and future-summed targets.

Suggested design:

- model: OLMo-3-32B-Think dev anchor;
- layers: 24, 32, 40, plus one late control;
- prompts: at least 12 from four families, none eligible for confirmation;
- positions: final prompt position, one bridge-support position, one neutral position;
- directions per cell: 8 random, 4 mean-J selected, 4 logit-aligned, 4 orthogonal controls;
- scales: one small measurable scale and one intervention-relevant scale;
- target variants: present position, future sum, paper aggregate;
- precision: measure input fidelity; use fp32 residual injection where feasible even if weights remain bf16.

### N3.2 Metrics

Report:

- response cosine;
- predicted/actual norm ratio;
- normalized error;
- token-row cosine;
- top-k candidate overlap;
- projector overlap and effective rank;
- predicted versus actual final-logit change;
- prompt, layer, position, and direction distributions, not only medians.

### N3.3 Dev decision rule

A context-J methods arm passes if, on held-out dev contexts:

- median response cosine improves by at least `0.20` over mean-J;
- reaches at least `0.80` on at least two of the three primary band layers;
- token-row or selected-set stability improves materially;
- gains are not confined to one prompt family;
- no outcome-selected candidate pool is used.

The exact thresholds may be adjusted before the run, but must be written into the config and committed first.

### N3.4 Interpretation

- If context J passes, add it as `contextJ_crossfit` in the scientific preregistration methods section.
- If it fails, keep mean J and report that context conditioning did not solve the mismatch under the tested estimator.
- In both cases, the `meanJ_paper` arm remains the behavioral replication primary.

**N3 stop rule:** do not refactor the entire campaign around context J before this gate. First let the directional evidence answer whether the dragon is real or merely a suspiciously dragon-shaped shrub.

## Stage N4: run G5 and expand the task bank

**Compute:** baseline model scoring plus CPU audits.  
**Tier:** dev.

### N4.1 Item manifest

Create one immutable row per item with:

```text
item_id
canonical_family
template_hash
source_pool
prompt
accepted_answers
canonical_answer
model_tokenizations
ground_truth_source
bridge_entity_or_state
counterfactual_prompt_and_answer
baseline_metrics_by_model
capability_flags_by_model
shortcut_flags
partition = UNASSIGNED
```

### N4.2 Two-hop gate

For every two-hop item:

- the bridge must not appear verbatim in the prompt unless the task definition allows it;
- direct answer-only shortcuts must be audited;
- a bridge swap or counterfactual must change the target in the expected direction;
- the model must answer the baseline item above a frozen capability threshold;
- relation/template family must be explicit.

### N4.3 Hard one-hop expansion

Expand until family-disjoint partitioning is feasible. Prefer productive shapes already identified by the pilot, but add new relation families rather than merely extending existing labels.

### N4.4 Working-set task gate

Build C1 tasks from synthetic generators with exact answers and independently variable:

- load;
- chain depth;
- delay;
- distractor count;
- surface length;
- answer vocabulary.

Generate train/dev templates for debugging and untouched confirmatory templates for the later split. Test for shortcuts by permuting labels and entity names.

### N4.5 Released task gate

Implement at least probe-swap, selectivity, capacity, directed modulation, and dual-task under frozen adaptation rules. Record any model-specific premise failure rather than silently changing the task.

**N4 gate:** G5 passes only when the immutable manifest and audit report are reproducible and every primary cohort meets its capability and family floor.

## Stage N5: prepare the binding preregistration candidate

**Compute:** CPU only.  
**Output:** `SCIENTIFIC_PREREGISTRATION_CANDIDATE.md`, not yet binding.

The candidate must include:

- the resolved D1-D6 decisions;
- exact corrected pilot evidence IDs used for design;
- exact primary hypotheses and analysis table;
- final sample sizes from corrected G6;
- primary matched control definition;
- mean-J primary and context-J methods-arm decision;
- model revisions and lens hashes;
- task-bank hash;
- family-level partition algorithm;
- tail threshold and sensitivity thresholds;
- exclusions;
- rerun policy;
- multiplicity families;
- mixed-model formula and fallback;
- TOST procedure;
- independent reproduction plan;
- prohibited analyses and claim wording.

Then stop and request user approval.

After approval, perform one dedicated freeze commit that:

1. renames the candidate to `SCIENTIFIC_PREREGISTRATION.md`;
2. runs the family-level deterministic partition;
3. stores only hashes and partition assignments, not intervention outcomes;
4. tags the commit `jspace-part2-confirmatory-freeze-v1`.

No other code or prose change belongs in that commit.

## Stage N6: run the confirmatory protected-dynamic study

### N6.1 Model order

Run complete model cells, not a thin smear across every checkpoint.

1. `allenai/Olmo-3.1-32B-Think`, own finalized lens;
2. `allenai/Olmo-3.1-32B-Instruct`, own finalized lens;
3. `Qwen/Qwen3.6-27B`, published/finalized lens and frozen raw-completion protocol;
4. optional base/older Think lineage points only after the primary pair is complete.

The current four-model pilot ladder does not include the actual primary `Olmo-3.1-32B-Think` endpoint. Do not mistake the historical `Olmo-3-32B-Think` pilot for the primary pair.

### N6.2 Confirmatory conditions

Minimum primary grid:

- `baseline`;
- `meanJ_protected_paper_literal` or the validated equivalent;
- `primary_geometry_matched_control`;
- `meanJ_unprotected` as diagnostic, not primary;
- `logit_protected` as prespecified secondary;
- `contextJ_protected` only if N3 passed.

Additional controls may run after the primary cells, but do not change the primary control after outcomes are seen.

### N6.3 Tasks

- released/validated two-hop factual set;
- difficulty-matched hard one-hop set;
- prose NLL guard;
- optional exact synthetic load set in its separate family.

### N6.4 Execution discipline

- randomize condition order within item or counterbalance blocks;
- checkpoint per item batch;
- emit one immutable per-item parquet per model/task;
- log requested rank, effective rank, selected IDs, strings, scores, singular values, energy, phase, and protection overlap;
- preserve clean and ablated outputs for a blinded audit sample;
- do not aggregate during the GPU run except for progress diagnostics;
- run the locked analysis from raw rows after the final cell banks.

### N6.5 Stop rules

Stop and investigate before proceeding to another model if:

- baseline capability differs from the frozen gate;
- protection invariant fails once;
- effective rank differs materially across primary conditions;
- primary control energy/spectrum matching fails tolerance;
- more than the preregistered fraction of items requires exclusion;
- payload hashes or model revisions disagree with the manifest;
- deterministic rerun of a sentinel item differs beyond tolerance.

## Stage N7: capacity and load studies

Run corrected R2 on held-out descriptive prompts and C1 on the frozen load set after the primary protected grid is banked. This prevents a long descriptive job from consuming the decisive causal window.

For capacity, report:

- occupancy distribution by layer;
- right-censoring;
- centered excess share with CI;
- random-dictionary seed sensitivity;
- solver sensitivity;
- fit-size and independent-lens sensitivity;
- model-relative depth and independently identified band.

For load, report the full interaction surface, not a cherry-picked high-load cell.

## Stage N8: independent rerun and paper assembly

Before release:

1. reproduce the primary table from a clean VM at the frozen commit;
2. preferably have a separate agent/session execute the registered commands without the exploratory narrative in context;
3. rerun the primary analysis from raw parquets;
4. verify every figure is a pure function of registered data;
5. compare confirmatory and replication partitions;
6. complete a claim-to-artifact ledger;
7. create a machine-readable release manifest;
8. freeze the handout, report, paper draft, and environment lock under a release tag.

---

# 8. Code change map

The next agent should add or revise the following components.

## 8.1 New or replaced modules

```text
interpretability/jspaces/phases/phase2/
  jspace_part2/
    paths.py                     logical artifact URI resolver
    registry.py                  schema-v2 event registry
    manifest.py                  model/lens/input/environment manifests
    family.py                    canonical family maps and audits
    stats.py                     locked paired, hurdle, mixed-model, TOST tools
    protected_dynamic_v2.py      corrected per-position and phase-aware assay
    occupancy_v2.py              corrected solver + centered excess variance
    partition.py                 deterministic family-level freeze split
    experiments/
      repair_pilot_families.py
      repair_r2_capacity.py
      h7_context_j.py
      g5_task_gate.py
      confirmatory_protected_grid.py
      confirmatory_analysis.py
      independent_repro.py
  schemas/
    evidence_event.schema.json
    result_envelope.schema.json
    item_manifest.schema.json
    run_manifest.schema.json
  data/
    probe_swap_family_map.json
  tests/
    test_registry_v2.py
    test_repro_fresh.py
    test_protected_generation_v2.py
    test_family_map.py
    test_occupancy_v2.py
    test_partition.py
    test_stats_v2.py
```

Keep old modules for historical evidence reproduction. New evidence must use the v2 modules.

## 8.2 Canonical family assertion

```python
def attach_family(rows: pd.DataFrame, family_map: pd.DataFrame) -> pd.DataFrame:
    out = rows.merge(
        family_map[["item_id", "canonical_family", "template_hash"]],
        on="item_id",
        how="left",
        validate="many_to_one",
    )
    if out["canonical_family"].isna().any():
        missing = out.loc[out["canonical_family"].isna(), "item_id"].unique()
        raise ValueError(f"missing canonical family for {missing[:10]!r}")
    return out
```

No analysis function should accept an unvalidated free-form family column.

## 8.3 Row-wise protected selection

Conceptually:

```python
scores = ...                              # [positions, vocab]
valid = scores > 0
valid.scatter_(1, protected_ids, False)
masked = scores.masked_fill(~valid, -torch.inf)

# Keep k per row, not min(k) over all rows.
values, ids = masked.topk(k, dim=1)
selected_valid = torch.isfinite(values)

# Build each row's rank-safe basis from only its valid selected rows.
# Batch by achieved count where useful, but never shrink unrelated rows.
```

## 8.4 Centered excess variance

The final implementation should expose every intermediate:

```python
Hc = H - H.mean(dim=0, keepdim=True)
Rc_j = R_j - R_j.mean(dim=0, keepdim=True)
Rc_r = R_r - R_r.mean(dim=0, keepdim=True)

den = (Hc.square()).sum()
r2_j = 1.0 - ((Hc - Rc_j).square()).sum() / den
r2_r = 1.0 - ((Hc - Rc_r).square()).sum() / den
excess = r2_j - r2_r
```

The report must not call a raw-energy ratio “variance explained.”

## 8.5 Registry lookup

```python
def resolve_evidence(events, evidence_id):
    created = [e for e in events
               if e["event"] == "evidence_created"
               and e["evidence_id"] == evidence_id]
    if len(created) != 1:
        raise RegistryError(...)
    record = dict(created[0])
    record["status_events"] = [e for e in events
                               if e.get("evidence_id") == evidence_id
                               and e["event"] != "evidence_created"]
    return record
```

Supersession is metadata about an evidence item, never a replacement for its creation record.

## 8.6 Deterministic family split

```python
rng = np.random.default_rng(frozen_seed)
families = stratified_family_table.sort_values("canonical_family")
# Allocate whole families while balancing difficulty and answer-length strata.
# Hash the exact input table, algorithm version, seed, and output manifests.
```

The code must prove that no canonical family appears in both partitions.

---

# 9. Statistics implementation requirements

## 9.1 Define the weighting estimand

The current bootstrap concatenates all items from resampled families, so larger families receive more weight. Decide whether the scientific population is:

- a random item from the authored bank; or
- a random relation family, then a random item within family.

For generalization across task families, equal-family weighting is usually the more defensible primary. Report item-weighted estimates as sensitivity.

## 9.2 Paired tail-rate inference

The primary binary effect is paired within item:

```text
hit_J = 1[delta_J < -1]
hit_C = 1[delta_control < -1]
paired_difference = mean(hit_J - hit_C)
```

Use a family bootstrap or a mixed-effects logistic model that preserves the pairing. Do not infer specificity from separate arm confidence intervals.

## 9.3 Mixed model

A reasonable frozen starting point for HP1 is:

```text
delta ~ model * task + baseline_accessibility + answer_token_count
        + (1 | canonical_family)
```

If item IDs are shared across models, include an item random intercept or a paired residual structure. Specify optimizer and fallback before outcomes.

## 9.4 Hurdle secondary

```text
Pr(tail) ~ model * task + ...
-magnitude | tail ~ model * task + ...
```

This separates how often the effect appears from how severe it is when it appears.

## 9.5 Equivalence

Implement real TOST and return:

- lower one-sided p-value;
- upper one-sided p-value;
- 90% CI;
- SESOI;
- equivalence decision.

Retire `equivalence_from_interval` as a generic function unless it explicitly receives the CI level and validates that the interval corresponds to the intended alpha.

## 9.6 Multiplicity

Name the exact primary family. Do not say “Holm over five” without enumerating the five tests. Secondary threshold curves, model ladders, layer sweeps, and control families should use their own declared FDR family or remain descriptive.

---

# 10. Model-specific instructions

## 10.1 OLMo

- Fit or validate an own lens for every confirmatory checkpoint.
- The primary comparison is 3.1 Think versus 3.1 Instruct.
- The base and historical 3.0 Think cells remain valuable lineage moderators but should not consume the primary window first.
- Record the exact training-lineage wording as a natural experiment, not a one-variable intervention.
- Run context-J methods work on dev OLMo only before deciding whether it generalizes.

## 10.2 Qwen

- Raw completion is appropriate for the cross-model protected-grid comparison if frozen identically.
- Official thinking-on/off is a chat-generation experiment and cannot use the current teacher-forced chat endpoint.
- Keep Qwen as an external validation model, not part of a claimed monotonic OLMo post-training sequence.
- Use the hard one-hop intersection cohort to remove the ceiling confound.

## 10.3 Gemma

- Keep it in the methods section.
- Use precise wording about finite-scale first-order approximation failure.
- Do not build causal J dictionaries from a fixed `W * gain` approximation without differentiating through the actual norm and softcap.
- Do not spend confirmatory budget on Gemma unless a separate, preregistered nonlinear-method project is approved.

---

# 11. Report and handout cleanup

The living report currently contains chronological strata that disagree with each other. For example, earlier sections say a Gemma fit is queued while later sections report its completion. The handout also combines results from different checkpoint moments.

Refactor reporting around a generated campaign state:

```json
{
  "current_commit": "...",
  "era": "pilot_repair_corrections",
  "gates": {"G0":"reopened", "G1":"reopened", ...},
  "active_preregistration": null,
  "latest_evidence": {...},
  "withdrawn_evidence": [...],
  "superseded_evidence": [...]
}
```

Every report build should:

- print branch, commit, generated time, and evidence cutoff;
- show a large `PILOT` or `CONFIRMATORY` label per figure;
- use only latest non-withdrawn evidence unless a historical comparison is explicit;
- include `n_items` and `n_families` together;
- distinguish mechanics controls from matched controls;
- distinguish raw energy from centered variance;
- distinguish teacher-forced likelihood from generated final-answer behavior;
- avoid “confirmed,” “resolved,” or “paper-faithful” unless the corresponding gate is actually passed.

---

# 12. Claim language after this review

## Allowed now

- Output protection removes most of the median unprotected live-ablation effect on the tested pilot sets.
- A protected J-specific damage tail remains in pilot data and is reproducible across two Instruct lens fits.
- The causal effect distribution is highly zero-inflated and heavy-tailed under the current pilot assay.
- Qwen's pilot task contrast is more forward-dissociated than the tested OLMo pilot contrasts.
- The current average-J estimator predicts only part of the measured aggregate OLMo transport response in the paper band.
- At tested finite perturbation scales, Gemma's source-to-target map is less well approximated by a linear response than OLMo's.
- The current J-lens recipe does not provide a useful mid-band Gemma readout.

## Not allowed yet

- The paper's causal result is confirmed on Qwen.
- The paper is refuted on OLMo.
- Post-training monotonically creates the workspace dissociation.
- OLMo has occupancy exactly 2 and Qwen exactly 3-4 under a paper-identical estimator.
- Gemma violates differentiability or cannot be modeled by any Jacobian.
- The protected random arm is energy- or geometry-matched.
- The current Part 2 package is fully reproducible from a clean VM.
- The scientific preregistration is ready to bind.

---

# 13. Decision tree for the eventual paper

## Outcome A: corrected pilots and confirmation preserve the tail and model interaction

Paper center:

> Output-protected J-space interventions reveal a sparse, model-dependent internal-content tail whose task selectivity changes with post-training, while average-J occupancy remains low in the tested open models.

Methods contribution:

- exact protection;
- geometry controls;
- tail-aware inference;
- mean-J versus context-J comparison.

## Outcome B: context J materially changes the causal result

Paper center:

> Corpus-averaged Jacobian lenses can understate or distort context-specific causal directions; context-conditioned transport changes the inferred J-space effect.

This becomes a methods paper with the model matrix as evidence.

## Outcome C: corrected clustering or controls remove the protected tail

Paper center:

> A rigorous open-model audit finds that the apparent J-specific causal tail is explained by family structure or intervention geometry, while readout phenomena remain.

This is still a useful boundary-of-generalization result.

## Outcome D: load tasks reveal a static or persistence-selected effect

Paper center:

> Workspace-like causal privilege is demand-dependent and emerges only when a controlled working set is engaged.

Characterize that positive before adding further architectures.

## Outcome E: all clean causal effects are equivalent to controls

Paper center:

> Descriptive J-space readouts transfer, but causal privilege does not under a preregistered, power-calibrated open-model assay with positive controls.

A rigorous null is publishable when the assay and equivalence test are credible.

---

# 14. Concrete deliverables for the next agent

The next agent should stop only after producing all of the following, or after documenting a hard blocker:

1. `reports/PILOT_SNAPSHOT_VM7.json` and release tag.
2. registry schema v2 and migrated snapshot.
3. a genuinely passing fresh-VM repro demonstration.
4. corrected protected generation module and tests.
5. canonical probe-swap family map and audit.
6. superseding R7 clustered intervals.
7. superseding tail-rate analysis.
8. superseding G6 power analysis.
9. corrected R2 solver and capacity artifacts.
10. D2 context-J methods report.
11. G5 item/task audit.
12. expanded family-disjoint item bank.
13. `SCIENTIFIC_PREREGISTRATION_CANDIDATE.md` with all decisions resolved.
14. regenerated `REPORT_PART2.md` and handout clearly labeled pilot.
15. a short `READY_FOR_FREEZE.md` listing every gate, evidence ID, and pass condition.

The agent must then stop before the binding freeze commit unless the user has explicitly approved the candidate.

---

# 15. Priority order under limited compute

Use this order. Complete each stage rather than sprinkling half-results across the map.

1. CPU package/repro fixes.
2. Family-map repair and statistical recomputation.
3. R2 correction and small reruns.
4. D2 context-J study.
5. G5 task gate.
6. Item-bank expansion and partition readiness.
7. Preregistration candidate.
8. User-approved freeze.
9. OLMo 3.1 primary pair.
10. Qwen validation.
11. Capacity and load sub-studies.
12. older lineage moderators.
13. independent rerun.
14. paper.

Do not spend another mainline GPU block on Gemma. Do not run more exploratory ladder checkpoints before the primary pair. Do not refit 250/500-prompt lenses merely because the old plan listed them; the independent 120-prompt Instruct fits already show high stability, and D2 is now the sharper estimator question. Fit-size scaling can return if corrected results show instability.

---

# 16. Final instruction to the coding agent

This branch has done the hard, rare thing: it has made its own attractive story less comfortable whenever the instruments demanded it. Continue in that spirit.

Do not optimize for preserving Figure 6. Optimize for making whatever replaces Figure 6 impossible to dismiss.

The immediate job is not to collect one more dramatic bar. It is to turn the current pilot into a sealed assay:

- correct the family units;
- correct the capacity estimand;
- correct generation protection;
- make reproduction real;
- determine what the averaged Jacobian loses;
- certify the tasks;
- freeze the endpoints and partitions;
- then run the primary pair once, cleanly.

A positive, a null, or a reversal can all become the paper. An ambiguous assay cannot.
