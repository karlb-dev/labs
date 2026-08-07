# jspace_part2_plan1_addendum.md: Forensic Review, Assay Repair, and a Publication-Grade Part 2

**Review target:** [`karlb-dev/labs@4097c44713d0084d4da3d3c084e79aed2068c740`](https://github.com/karlb-dev/labs/commit/4097c44713d0084d4da3d3c084e79aed2068c740)  
**Primary plan reviewed:** `jspace_interp_part2_plan1.md`  
**Purpose:** convert an ambitious exploratory replication into a confirmatory, reproducible research program whose positive, negative, or mixed result can survive serious peer review.

> **Paste-line for the coding/research agent**  
> Read this addendum before executing `jspace_interp_part2_plan1.md`. Treat the merged Lab 37 commit and saved Drive runs as exploratory evidence and reusable engineering assets, not as locked scientific conclusions. Begin with the Stage 0 and Stage 1 assay-repair gates in this document. Do not launch the cross-model matrix until the paper-faithful intervention, paper-faithful capacity metric, paired statistics, phase-specific hooks, and raw per-item logging pass their validation tests. Preserve all existing artifacts, create a new `part2_<date>/` run root, and distinguish exploratory analyses from preregistered confirmatory analyses in filenames, manifests, figures, and prose.

---

## 0. Executive assessment

The merged lab is a **strong exploratory mechanistic-interpretability project**. It contains unusually good engineering instincts for a first campaign: resumable GPU work, incremental checkpointing, explicit falsifiers, control conditions, artifact-backed reporting, and a willingness to report nulls. It also discovered several genuinely interesting phenomena worth pursuing, particularly:

1. a J-lens readout advantage over the logit lens on some latent-content probes;
2. a large internal difference between OLMo and Qwen under the lab's current sparse-decomposition metric;
3. a reproducible effect from prompt-selected J-aligned interventions on factual-answer probabilities;
4. rich temporal traces relating J-lens readouts to generated reasoning tokens;
5. several negative results that sharply narrowed the original search space.

However, the current commit is **not yet publication-grade evidence for or against the paper's central global-workspace claims**. Three issues are especially important:

### 0.1 The primary causal intervention is not the paper's intervention

The paper's broad J-space ablation dynamically removes the ten most active J-lens vectors at each token position, but explicitly protects any token direction that is among the clean forward pass's top-ten output tokens. That safeguard is intended to avoid deleting the token the model is presently trying to emit. The lab's `jspace_dyn10` implementation does not perform this clean-output protection. Its asterisk collapse is therefore not a faithful reproduction of the published intervention, and it cannot yet support the proposed conclusion that OLMo's or Qwen's verbalizable space is occupied by imminent output rather than deliberative content.

The lab's static shared-span intervention is a useful novel test, but it is not the paper's headline intervention either. A static-span null can show that one corpus-level span is not causally sufficient at the tested doses. It cannot, by itself, establish that the paper's position-specific, live J-space dissociation fails to transfer.

### 0.2 The reported “capacity” numbers are not commensurate with the paper's capacity definition

The paper defines occupancy as the value of `K` where the marginal reconstruction improvement from a sparse, non-negative J dictionary falls below the marginal improvement from an equal-size random control. It then reports variance explained **in excess of the matched random dictionary**, evaluated at the median occupancy. The lab instead reports thresholded coefficient counts and a raw reconstruction-to-activation variance ratio at a fixed pursuit budget.

Those are reasonable exploratory metrics, but they are different estimands. Therefore:

- “OLMo has approximately six active concepts while Claude has approximately twenty-five” is not yet an apples-to-apples comparison;
- “OLMo's workspace is ten times thinner than the paper's” is not currently supported;
- Qwen's larger value under the same lab code path is evidence of an internal cross-model difference under that code path, but not yet evidence that Qwen is in the paper's occupancy or variance range.

### 0.3 Several decisive effects lack the controls, endpoints, and paired statistics required for causal interpretation

The frozen per-item intervention is promising, but its J and random controls are not matched per item for removed energy, effective rank, dictionary coherence, or singular spectrum. QR is used without a numerical-rank test, so near-duplicate selected directions can silently add arbitrary orthogonal columns. The hooks also affect prompt prefill and decoding together, although the prose often describes generation-only application.

The chain-of-thought lead and rescue results require a similar downgrade. The original mid-band lead analysis used a selected subset of saved traces, and the rescue headline counts the answer appearing anywhere within up to 400 generated tokens. Most runs did not close `</think>`, and final-answer accuracy after `</think>` was much lower. This is interesting evidence that extended generation often re-expresses deleted content, but it is not yet a clean demonstration that externalized reasoning rescues task performance.

### 0.4 Bottom-line verdict

The current lab does **not** prove the paper, disprove the paper, or establish a model-family-specific replacement theory. It does something more useful than a failed replication: it identifies a cluster of plausible phenomena, exposes instrument fragility, and supplies a mature codebase from which a decisive study can be built.

The best Part 2 is therefore not “run more models with the existing assay.” It is:

1. repair and validate the assay against the paper's exact definitions;
2. re-establish which Part 1 findings survive the repaired assay;
3. preregister the competing hypotheses and primary endpoints;
4. run the OLMo lineage as the main causal comparison;
5. use Qwen and Gemma as deliberately different secondary contrasts;
6. reserve a final untouched item set and independent rerun for confirmation.

That ordering is slower for the first few hours and dramatically faster scientifically. Otherwise the campaign risks producing a beautiful model zoo full of differently colored ambiguity.

---

## 1. Scope, source map, and what was actually reviewed

### 1.1 Repository scope

The referenced SHA is the squashed merge commit for the complete Lab 37 campaign, not a narrow initial experiment. It includes the v1 lens and descriptive work, the v2 instrument audit, the late-band lens, the frozen intervention, the chain-of-thought analyses, the Qwen leg, robustness runs, final falsifiers, report generation, figures, and the student-facing lab materials.

This distinction matters because the commit history no longer separates the chronological state at which each claim was first formed. This review treats the merged artifact as the research object and reconstructs chronology only where the code, reports, plans, and logs permit it.

### 1.2 Primary sources

The review is grounded in:

- the target commit and every committed script under [`interpretability/jspaces/phases/phase1/code`](https://github.com/karlb-dev/labs/tree/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code);
- [`REPORT_v2.md`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/report/REPORT_v2.md);
- [`lab37_jspace_workspace.md`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/labs/lab37_jspace_workspace.md);
- committed result summaries under [`interpretability/jspaces/phases/phase1/results`](https://github.com/karlb-dev/labs/tree/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/results);
- the original prompt, v2 delta prompt, three-hour execution plan, Part 2 plan, and planning chat supplied with this review;
- the paper, its released prompt descriptions, and the reference `anthropics/jacobian-lens` implementation;
- Neel Nanda's independent Qwen replication and methodological discussion;
- official model cards for the OLMo, Qwen, and Gemma checkpoints proposed for Part 2.

### 1.3 Limits of this review

This was a forensic review of committed code and available artifacts. It did not rerun the 32B GPU experiments. The commit does not contain every lens checkpoint, frozen prompt file, environment file, full trace, or per-item score needed for an independent numerical replay. Several conclusions below are therefore code-level or design-level findings. Where a result might survive after correction, this document says so explicitly rather than treating a detected flaw as proof that the effect is false.

### 1.4 Evidence vocabulary used below

To prevent verbal inflation, this addendum uses four labels:

- **Established in this lab:** directly supported by a valid comparison and an auditable artifact under the stated scope.
- **Provisional:** a real measured pattern, but one or more material controls, endpoints, or replication steps remain.
- **Exploratory:** useful for hypothesis generation, but selected or developed after inspecting the data.
- **Not currently identified:** the implemented measurement does not estimate the claimed quantity or reproduce the claimed protocol.

A null result is never upgraded merely because two confidence intervals overlap. A positive result is never upgraded merely because a control's point estimate is smaller.

---

## 2. What the merged lab actually establishes today

The most defensible current summary is narrower than `REPORT_v2.md`, but it remains scientifically useful.

### 2.1 Established or close to established

1. **A usable averaged-Jacobian readout was fitted for OLMo-3-32B-Think.** The lens passes nontrivial qualitative and quantitative sanity checks, and in some probe sets it ranks latent answers more effectively than the plain logit lens.
2. **The lab's original non-J PCA damage was largely an activation-energy confound.** Matching aggregate removed energy substantially reduced the apparent difference among static conditions.
3. **Unprotected live J-direction removal can catastrophically disrupt generation while matched random live removal does not.** This is a valid observation about that implemented intervention, although it is not a paper-faithful ablation.
4. **Prompt-selected J-aligned projection changes factual-answer probabilities more than the implemented random-dictionary control on OLMo and Qwen.** This is a reproducible provisional effect that deserves better controls.
5. **The current OLMo and Qwen pipelines produce very different sparse-reconstruction summaries.** The cross-model difference is real under the lab's current definitions, even though it cannot yet be labeled a paper-range capacity difference.
6. **The saved temporal readouts often surface answer-related token directions before those strings appear in generated reasoning.** The all-item late-band trace is the stronger version of this observation; formal lead-time inference still needs a preregistered detector and matched foil analysis on the same trace population.

### 2.2 Provisional and worth preserving

1. Frozen J-aligned intervention as a content-channel perturbation.
2. A difference between one-hop and two-hop sensitivity on Qwen.
3. Increased answer visibility during extended thinking after a frozen intervention.
4. Differences in J-lens geometry across post-training regimes.
5. A potential distinction between output-aligned and workspace-like J-space content.

### 2.3 Not currently identified

1. Failure of the paper's output-protected dynamic J-space ablation on OLMo or Qwen.
2. A tenfold paper-comparable capacity deficit in OLMo.
3. A paper-comparable broadcast null.
4. Clean causal specificity of the frozen intervention.
5. Final-answer rescue by externalized reasoning.
6. A causal inverse relationship between output occupancy and workspace sensitivity.
7. Equivalence of the tested static interventions to zero effect.

---

## 3. Major strengths of Part 1

A rigorous addendum should preserve what was done well, because these are the bones of the publishable study.

### 3.1 Strong research-engineering discipline

The campaign was designed around fragile Colab sessions and expensive models, yet it produced resumable scripts, atomic JSON writes, chunked lens fitting, phase-level checkpoints, Drive mirroring, hardware audits, and CPU-only report regeneration. That is materially better than the usual one-notebook replication. Keep this architecture.

### 3.2 Explicit falsifiers and willingness to revise claims

The progression from unmatched PCA controls to energy matching, from live intervention to frozen selection, from uncalibrated lead to foil checks, and from a single model to Qwen shows excellent falsification instincts. The report also records flipped and sharpened claims rather than sanding away inconvenient results.

### 3.3 Useful separation of descriptive, causal, and temporal questions

The project did not confuse “the lens can read a concept” with “the concept is causally necessary” at the design level. It built separate sanity, descriptive, ablation, chain-of-thought, and reporting phases. The problem is not the decomposition; it is that some individual assays do not yet match their intended estimands.

### 3.4 Reusable multi-model harness

The Qwen adaptation demonstrates that the code can survive a different vocabulary size, model wrapper, and numerical regime. With a formal adapter interface and conformance tests, this can become a valuable open replication harness rather than a one-off lab.

### 3.5 Honest negative results

The static-span null, lack of a clean broadcast distinction under the current fan-out metric, weak eval-awareness result, and pre-CoT anticipation null were not hidden. Null-friendly reporting is an asset. Part 2 should add formal equivalence methods so that “honest null” can become “quantified evidence for a bounded null.”

---

## 4. Critical findings, ordered by publication risk

| Priority | Finding | Why it matters | Required action |
|---|---|---|---|
| Blocker | Live ablation omits the paper's clean-output top-ten protection | The asterisk collapse is not a replication of the paper's intervention | Implement exact output-protected dynamic ablation and validate it on a tiny model plus released tasks |
| Blocker | Capacity and variance metrics differ from the paper's definitions | The “six vs twenty-five” and “ten times thinner” claims are non-commensurate | Reimplement marginal-gain occupancy and random-adjusted variance at median occupancy |
| Blocker | Static span is treated as the paper's causal signature | The paper's broad ablation is dynamic and position-specific | Reframe static span as a novel auxiliary instrument, not the decisive replication test |
| Blocker | Causal inference uses separate condition bootstraps and CI overlap | Item pairing is discarded; nulls and differences are mischaracterized | Save per-item outcomes, use paired contrasts, cluster bootstrap, and equivalence tests |
| Blocker | Frozen controls do not match effective rank, energy, or geometry | The marquee content-specific effect has plausible geometric alternatives | Add rank-safe bases and rotated-J, shuffled-label, logit, and item-energy-matched controls |
| Blocker | CoT rescue endpoint is answer-anywhere with unmatched windows | Mentioning an answer during long generation is not recovered final performance | Make post-`</think>` final-answer correctness primary and equalize token/compute budgets |
| Major | Mid-band lead traces were outcome-selected | The 46-step estimate is biased by which traces were saved | Save all traces; rerun detector on a preregistered complete sample |
| Major | Hooks combine prompt prefill and decoding | “Generation-only” and “prompt-selected” interpretations are ambiguous | Factor intervention phase: prefill only, decode only, both, neither |
| Major | Broadcast assay is not the paper's assay | Weight-projection fan-out cannot confirm or refute paper broadcast | Implement MLP gain, attention OV gain/label preservation, and J-rotated control |
| Major | Cross-model “same harness” changes many variables | Differences can reflect wrapper, lens, metric, task, or numerical changes | Introduce model adapters, conformance tests, and identical primary estimands |
| Major | Raw per-item and provenance artifacts are incomplete in git | Independent reviewers cannot reproduce paired analyses | Commit or release manifests, prompt hashes, per-item parquet, and artifact checksums |
| Major | Lens transfer is proposed as a substitute for recipient fitting | A J-lens is a model-specific mean Jacobian | Treat transfer as a scientific outcome, but fit a canonical recipient lens for confirmatory work |
| Moderate | SQL `n=30` contains only three schemas | Template replication masquerades as independent sample size | Generate many independent schemas and cluster inference by schema family |
| Moderate | First-token answer metrics fail on multi-token answers | Readout and behavior can be mis-scored | Use full answer-sequence conditional log probability and normalized exact matching |
| Moderate | Random static doses are not nested | Apparent dose response includes direction redraw noise | Draw one random orthobasis per seed and use nested prefixes |
| Moderate | Resumable PCA moments are not actually resumable in OLMo scripts | A resumed run can produce partial-corpus covariance state | Persist mergeable streaming moments or restart the entire descriptive phase |

---

## 5. Detailed code and measurement review

### 5.1 Configuration, model loading, and provenance

Relevant files:

- [`sl1_common.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/sl1_common.py)
- [`s0_env_audit.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s0_env_audit.py)
- [`s2_corpus.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s2_corpus.py)

#### Good

- Run directories and artifact paths are centralized.
- Seeds are recorded and used consistently in many scripts.
- GPU, CUDA, Torch, Transformers, `jlens` revision, and `pip freeze` are collected at run time.
- Corpus content receives a SHA in the Drive artifact.
- Atomic JSON writes reduce corruption risk.

#### Problems

1. **Model, tokenizer, and dataset revisions are not pinned.** `from_pretrained(model_id)` and dataset loading by name are mutable. A rerun months later can silently fetch different code, tokenizer files, dataset snapshots, or model metadata.
2. **The committed repository does not contain the environment and prompt artifacts referenced by the report.** Reviewers can see scripts and aggregate results, but not necessarily the exact inputs and package state.
3. **Hard-coded Colab and Drive paths leak environment assumptions into scientific logic.** This makes local and independent reproduction brittle.
4. **Seed setting is not determinism.** CUDA kernels, generation libraries, model wrappers, and multiprocessing can still vary. For confirmatory greedy cells, deterministic algorithms should be enabled where supported and any unsupported operations logged.
5. **No artifact lineage graph exists.** A result JSON should identify the exact model revision, tokenizer revision, lens hash, prompt-set hash, code commit, environment lock hash, and upstream files that produced it.

#### Required Part 2 change

Create one immutable `run_manifest.json` at run launch and append phase records to it. A result file is invalid unless it points to that manifest and includes its own input hashes.

Recommended fields:

```json
{
  "study_id": "jspace-part2-confirmatory-v1",
  "code_commit": "<git sha>",
  "dirty_tree": false,
  "model": {
    "id": "allenai/Olmo-3.1-32B-Think",
    "revision": "<resolved sha>",
    "config_sha256": "...",
    "tokenizer_sha256": "..."
  },
  "lens": {
    "fit_id": "olmo31-think-wikitext-n500-seed0",
    "sha256": "...",
    "jlens_commit": "581d3986...",
    "source_layers": [20, 22],
    "target_layer": 63
  },
  "prompts": {
    "fit_sha256": "...",
    "evaluation_sha256": "...",
    "holdout_sha256": "..."
  },
  "environment": {
    "lock_sha256": "...",
    "container_digest": "...",
    "gpu": "NVIDIA H100 96GB",
    "driver": "..."
  }
}
```

### 5.2 Lens fitting

Relevant files:

- [`s3_fit_32b.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s3_fit_32b.py)
- [`s14_lateband_fit.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s14_lateband_fit.py)
- reference [`jlens/fitting.py`](https://github.com/anthropics/jacobian-lens/blob/581d3986/jlens/fitting.py)

#### Good

- Chunking 120 prompts into mergeable 30-prompt lenses is appropriate for interruption tolerance.
- The fit uses the released `jlens` engine rather than a local reimplementation.
- Source and target layers, sequence length, skipped positions, VRAM, and elapsed time are logged.
- The late-band fit addresses a real coverage concern.

#### Problems

1. **No held-out lens-quality curve is reported against fit size.** A usable lens at `n=120` is plausible, but “usable” is not equivalent to “stable enough for a causal null.”
2. **No between-fit uncertainty is measured.** One merged lens can hide substantial sampling variance across corpus draws.
3. **The late-band lens is not statistically independent of the mid-band lens.** It uses the same model, fitting corpus, recipe, target, and run context. It is a separate source-layer fit, which is useful, but calling it an independent lens overstates the evidence.
4. **Transfer across post-training variants is proposed as a compute shortcut.** Because the lens is an average Jacobian of a specific model, transfer can be measured but should not replace fitting the recipient model in the confirmatory analysis.
5. **Fit-size comparison by raw per-token cosine is insufficient.** Sign, local rotations, token aliases, near-duplicate directions, and layer-wise geometry can make naive cosine comparisons misleading.

#### Publication-grade lens study

For the primary OLMo lineage, fit at least:

- `n = 25, 50, 100, 250, 500`, with nested samples;
- three independent corpus draws at a strategically selected subset, for example `n = 100` and `n = 500`;
- WikiText-like and Dolma 3-like distributions;
- the same source layers and target definition across checkpoints.

Evaluate on untouched held-out corpora and task probes using:

1. normalized area under pass@k versus log k;
2. full-token-sequence latent-answer rank where applicable;
3. top-k token-set agreement;
4. dictionary Gram-matrix CKA or Procrustes-aligned geometry;
5. J-space occupancy and variance under the paper-faithful estimator;
6. selected-direction stability and projector principal angles;
7. causal effect stability on a small locked calibration set.

Do not choose the canonical `n` solely by whether the headline effect increases. Choose the smallest `n` whose held-out metrics are inside preregistered equivalence margins relative to `n=500`.

### 5.3 Lens sanity checks

Relevant file: [`s4_lens_sanity.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s4_lens_sanity.py)

#### Good

- The script compares J-lens and logit-lens ranks.
- It includes a multihop bridge-entity evaluation rather than only hand-picked probes.
- It records per-item values.

#### Problems

1. Taking the best rank across many layers and token variants creates a large multiple-opportunity advantage.
2. The pass gate is permissive and partly defined relative to the logit lens, not to an externally meaningful quality floor.
3. Multi-token answers are represented by first-token variants, which can reward generic prefixes or fragments.
4. There is no negative-control lens, such as a layer-shuffled Jacobian, random orthogonal map, or token-label permutation.
5. The same probe families are used repeatedly throughout development, inviting calibration overfitting.

#### Required gates

A lens passes only if it meets all of the following on a hidden validation set:

- J-lens beats logit lens on a preregistered normalized pass@k AUC by a minimum effect size;
- answer ranks beat a label-shuffled lens and a layer-shuffled lens;
- results are stable across at least two lens fits or fit samples;
- qualitative inspection of a random, not cherry-picked, sample shows an acceptable false-positive rate under a written rubric;
- the recipient-fitted lens outperforms or is equivalent to a transferred donor lens before transfer is used for any substantive claim.

### 5.4 Sparse decomposition and descriptive geometry

Relevant files:

- [`s5_descriptive.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s5_descriptive.py)
- [`s15_lateband_readouts.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s15_lateband_readouts.py)
- Qwen implementation in [`s18_qwen_instruments.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s18_qwen_instruments.py)

#### Critical numerical issues

1. **Resume state is incomplete.** Aggregate JSON records completed batches, but PCA moments live only in RAM in the OLMo path. If a process resumes after partial completion, the final covariance can describe only the newly processed suffix while the other aggregates include all batches.
2. **Centered variance is accumulated using each batch's local mean.** Summing within-batch centered sums of squares omits between-batch mean variation. The correct global centered second moment requires a mergeable Welford or Chan update.
3. **The 90% reconstruction-energy statistic treats coefficient squares as additive.** In an overcomplete, correlated dictionary, cross-terms matter. Cumulative coefficient energy is not equal to reconstruction energy.
4. **Gradient pursuit differs across OLMo and Qwen.** The Qwen path changes precision and step-size behavior after numerical instability. That may be a necessary fix, but it breaks the claim that the capacity values were produced by an identical estimator unless OLMo is recomputed using the final validated algorithm.
5. **The reported active-count threshold is not the paper's occupancy estimator.** It should be retained only as a secondary descriptive diagnostic.
6. **Raw reconstruction variance is not adjusted by a matched random dictionary.** It therefore cannot be placed directly beside the paper's excess-variance result.

#### Correct estimator

At each evaluated activation `h` and layer:

1. solve a sparse non-negative approximation over the J dictionary for `K = 0..K_max`;
2. solve the identical approximation over multiple equal-size random dictionaries;
3. calculate marginal reconstruction improvement `Delta_J(K)` and the random-control distribution `Delta_R(K)`;
4. define occupancy using a preregistered crossing rule, matching the paper as closely as released details permit;
5. at the layer's median occupancy, compute globally centered variance explained by J minus the matched random control;
6. report distributions over positions, prompts, tasks, layers, lens fits, and random seeds.

The active coefficient threshold and fixed-`K` reconstruction curve can remain as secondary robustness plots, but they must not be called the paper's capacity metric.

#### Solver validation

Before running a 32B model, construct synthetic mixtures where the true sparse support, coherence, noise, and coefficient sign are known. Compare:

- the current gradient pursuit;
- non-negative orthogonal matching pursuit;
- non-negative least squares on the selected support;
- a high-accuracy reference solver on small dimensions.

Acceptance criteria should cover support recovery, reconstruction error, monotonicity, duplicate atoms, high-coherence dictionaries, half-precision behavior, and deterministic reproducibility.

### 5.5 Broadcast

Relevant files:

- [`s6_broadcast.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s6_broadcast.py)
- final fan-out block in [`s24_final_falsifiers.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s24_final_falsifiers.py)

The current metric asks how many later weight matrices have projection norms more than three random-direction standard deviations above a baseline. This is a useful exploratory weight-alignment score. It is not the paper's broadcast assay.

The paper separates broadcast across depth and across token positions:

- depth: nonlinear MLP gain on a direction, normalized to isotropic random directions;
- tokens: attention-head OV gain plus label preservation;
- controls include a fixed orthogonal rotation of the J dictionary that preserves spectrum and pairwise geometry;
- causal follow-up ablates identified broadcast heads and measures workspace-content and behavioral disruption.

The current lab:

- does not evaluate nonlinear MLP responses;
- does not measure attention-head label preservation;
- does not use a geometry-preserving J-rotated control;
- treats many correlated directions as independent samples in significance tests;
- cannot distinguish a direction's large norm or variance from selective broadcast;
- therefore cannot support the conclusion that the paper's broadcast signature is absent or nonspecific.

#### Part 2 replacement

Implement the paper's two structural metrics as the confirmatory analysis. Keep the current fan-out score as an exploratory appendix metric. Add:

1. isotropic random directions;
2. J-rotated directions preserving the J dictionary's Gram geometry;
3. matched high-variance residual PCs;
4. logit-lens directions;
5. layer-matched MLP-output and SAE directions where available;
6. five random seeds for direction/control sampling;
7. model- and layer-clustered uncertainty, not a direction-level IID test.

If broadcast heads can be identified, conduct a preregistered causal follow-up with equal-size, layer-matched random head sets across at least five seeds.

### 5.6 Static and dynamic ablation

Relevant files:

- [`s7_ablation.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s7_ablation.py)
- [`s11_energy_match.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s11_energy_match.py)

#### What the current static result means

The aggregate-energy-matched static intervention is useful evidence that the selected corpus-level J span, at `k <= 40` and the tested layer band, does not produce a large effect on the current task set. It should be stated that narrowly.

It does not show:

- equivalence to zero at a peer-review-relevant margin;
- absence of effects at higher doses or different selection rules;
- absence of effects under high working-memory load;
- failure of the paper's dynamic, output-protected intervention;
- absence of a nonlinear or prompt-specific workspace manifold.

#### Method issues

1. Energy is calibrated on a descriptive corpus, not on the evaluated items or decode positions.
2. J and control bases can have different effective ranks.
3. QR is used without rank truncation.
4. Static random doses are not guaranteed to be nested prefixes of one seed-specific basis in every path.
5. Baseline and some J cells are imported from an earlier file rather than rerun in a single randomized block.
6. Hook application covers both prefill and decoding.
7. The statistical analysis bootstraps each condition separately instead of bootstrapping paired item differences.
8. “Every CI contains baseline” is used as evidence of a null.

#### Required factorial

For every decisive intervention, run a `2 x 2` phase factorial:

| Prefill | Decode | Interpretation |
|---|---|---|
| off | off | baseline |
| on | off | alter prompt encoding or retrieval only |
| off | on | alter recurrent decode-time computation only |
| on | on | total intervention used by many current scripts |

For teacher-forced log-probability outcomes, define precisely whether intervention begins before the answer position and whether cached keys and values were computed under the intervention.

### 5.7 Frozen per-item intervention

Relevant files:

- [`s12_frozen_ablation.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s12_frozen_ablation.py)
- pool-size falsifier in [`s24_final_falsifiers.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s24_final_falsifiers.py)

This is the most promising novel instrument in the lab, but the report's “control-clean” label is premature.

#### Current alternatives that remain open

- J-selected rows may remove more prompt energy than random-selected rows.
- J rows may be more mutually coherent, causing a different effective projector.
- near-duplicate J rows may make raw QR return arbitrary completion directions;
- J selection has a much deeper candidate pool and a semantically structured extreme-value distribution;
- absolute correlations select directions that are strongly negative as well as strongly positive, although the J-space decomposition is defined non-negatively;
- J and random controls are not matched on singular spectrum or projector principal angles to the prompt activation subspace;
- prompt prefill is recomputed under the projector, so the intervention can erase the content before generation begins rather than remove a maintained channel during generation.

The vocab-sized random check is helpful, but it does not close these alternatives. It uses a smaller first-30 subset and compares against aggregate reference means rather than a paired baseline on those same items. Overlapping confidence intervals are not evidence of equivalence.

#### Required control family

For each item and layer, compare at least:

1. **J-selected:** top non-negative J activations under the locked selection rule.
2. **Isotropic random:** candidate pool matched in size, selected by the same rule.
3. **J-rotated:** one fixed orthogonal rotation preserving J dictionary spectrum and pairwise geometry.
4. **Label-shuffled J:** preserve vectors and activations, permute token labels before content targeting.
5. **Logit dictionary:** final-norm-adjusted unembedding rows.
6. **Matched non-J:** selected to match removed prompt energy, effective rank, and singular spectrum within tolerance.
7. **Counterfactual content:** J directions for a plausible but wrong intermediate or answer.
8. **Distractor content:** J directions active in a matched unrelated prompt.

Every projector must record:

- selected token IDs and decoded strings;
- raw scores and sign;
- singular values;
- numerical rank;
- removed energy on the clean prompt, answer position, and each decode step;
- overlap with clean top-output tokens;
- overlap across layers and lens fits;
- projector hash.

### 5.8 Chain-of-thought temporal analysis

Relevant files:

- [`s8_cot.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s8_cot.py)
- [`s8b_cot_lead.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s8b_cot_lead.py)
- [`s13_cot_foils.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s13_cot_foils.py)
- late trace phase in [`s15_lateband_readouts.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s15_lateband_readouts.py)

#### Lead-time result

The original mid-band trace saver kept the first ten items and additional items with at least two detected divergence events. The later lead estimator operates only on saved traces. This creates outcome-dependent sampling. The 46-step median should be treated as exploratory.

The late-band run appears to trace all items, which is substantially stronger, but it needs:

- the same foil detector on the same all-item population;
- uncertainty intervals and item-family clustering;
- a token-sequence-aware answer detector;
- prespecified layer aggregation rather than “any top-eight token in any read layer”;
- a baseline that accounts for repeated opportunities over long traces;
- separate analyses for cases where the answer is eventually correct versus incorrect.

#### Detector problems

- Multi-token concepts are reduced to first-token variants.
- Text detection removes spaces and uses substring matching.
- Any of several layers and many time steps can create a hit.
- Frequency matching is based on a tiny fitting corpus's surface words rather than tokenizer-level frequency or model probability.
- Foils are not consistently matched for token length, prior probability, semantic category, or answer morphology.

#### Better temporal endpoint

For a target concept represented by a token sequence or phrase template, estimate a calibrated evidence score at each step. Define onset using a threshold chosen on a separate calibration set to achieve a fixed false-positive rate against matched foils. Report:

- onset relative to first explicit textual commitment;
- area under the evidence curve before textual mention;
- probability the target outranks all matched foils;
- time-resolved target-versus-foil margin;
- correctness-stratified and task-stratified effects;
- uncertainty clustered by item family.

A hidden Markov or change-point model can be useful, but a simple preregistered thresholded margin is preferable to an ornate post hoc detector.

### 5.9 Chain-of-thought rescue

Relevant files:

- [`s16_cot_rescue.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s16_cot_rescue.py)
- filler falsifier in [`s24_final_falsifiers.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s24_final_falsifiers.py)

The current `0.23 -> 0.80` comparison should not be described as final-answer rescue:

- `0.80` is answer occurrence anywhere in a long trace;
- the post-`</think>` score is much lower;
- many outputs never close the think block;
- there is no same-run unablated thinking baseline in the capped grid;
- projectors are recomputed from a chat-rendered prompt rather than reusing exactly the no-think selection artifact;
- the real-CoT and filler arms use different answer-detection windows;
- the filler contains semantically reasoning-like language, so it is not a pure compute-only control.

#### Required rescue design

Use one immutable per-item selection artifact, then randomize generation arms:

1. no intervention, direct answer;
2. J intervention, direct answer;
3. matched control, direct answer;
4. no intervention, thinking;
5. J intervention, thinking;
6. matched control, thinking;
7. J intervention, semantically irrelevant natural text of matched length;
8. J intervention, shuffled valid reasoning tokens;
9. J intervention, externally supplied correct rationale;
10. J intervention, externally supplied plausible wrong rationale.

Primary endpoint: exact or rubric-graded final answer in a fixed post-reasoning response segment.  
Secondary endpoints: answer mention during reasoning, closure rate, reasoning length, answer log probability, re-retrieval of the deleted intermediate, and J-space re-entry.

Equalize the allowed total tokens and, where possible, the number of forward passes. Analyze rescue as a paired interaction:

`(J_think - J_direct) - (control_think - control_direct)`.

This separates a general benefit of longer generation from a selective ability of reasoning to compensate for the J intervention.

### 5.10 Eval-awareness

Relevant file: [`s10_evalaware.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s10_evalaware.py)

The current explicit prefix changes lexical content, pragmatic framing, expected response style, and likely output distribution. A dense vocabulary-space delta norm over twenty pairs can detect any consistent framing change. It is not a specific test of latent evaluation awareness.

The negative result is therefore weak evidence either way. Do not prioritize this in Part 2 until the main workspace assay is stable.

A serious version requires:

- implied test cues with no evaluation lexicon;
- matched non-evaluation institutional cues;
- cross-validated direction discovery and held-out behavioral endpoints;
- label permutation and prompt-template random effects;
- a behavioral task where evaluation framing demonstrably changes behavior;
- intervention strength calibrated on held-out prompts;
- correction for searching across layers and token labels.

### 5.11 Qwen cross-model leg

Relevant file: [`s18_qwen_instruments.py`](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code/scripts/s18_qwen_instruments.py)

The Qwen run is valuable as an engineering and exploratory extension. It is not a one-variable “same harness” test because it changes:

- model and tokenizer;
- lens source and fit corpus;
- all-layer versus subset-layer coverage;
- vocabulary size;
- random-dictionary size;
- numerical precision and pursuit update;
- sample sizes;
- prompt rendering and inference mode;
- baseline task difficulty.

Qwen 3.6 is a hybrid thinking model. Its official non-thinking mode is a chat-template or API configuration on the same checkpoint, with different recommended decoding settings. Raw completion prompts do not automatically constitute the official non-thinking condition.

For Part 2, split Qwen into two questions:

1. **Within-checkpoint inference-mode contrast:** identical weights and lens, official thinking enabled versus disabled. This tests whether externalizing reasoning during inference changes occupancy, intervention sensitivity, or temporal dynamics.
2. **Post-training or lineage contrast:** a base or separately post-trained checkpoint, when available and scientifically matched. This tests whether learned reasoning policy changes the internal workspace.

These contrasts must not be pooled under one “Qwen non-think sibling” label.

### 5.12 Statistical reporting and result storage

The committed causal JSONs usually retain aggregate means and separate bootstrap intervals, but not all per-item condition outcomes. This prevents the most informative analysis: paired within-item contrasts.

Required raw table schema:

| field | example |
|---|---|
| `study_id` | `part2-confirmatory-v1` |
| `model_id`, `revision` | exact checkpoint SHA |
| `lens_id`, `lens_hash` | immutable lens artifact |
| `item_id`, `family_id`, `template_id` | clustering units |
| `condition`, `phase`, `dose`, `seed` | intervention factors |
| `baseline_score`, `score` | paired outcome |
| `delta` | condition minus baseline |
| `answer_token_ids` | full sequence |
| `selected_direction_ids` | per layer |
| `effective_rank` | per layer |
| `removed_energy` | prompt and decode summaries |
| `output_protected_ids` | clean top-ten exclusion |
| `generation_text_path` | immutable raw output |
| `manifest_hash` | provenance |

Store this as Parquet or Arrow plus a human-readable sample. Aggregate JSONs become derived report artifacts, never the sole scientific record.

---

## 6. Claim-by-claim revision of `REPORT_v2.md`

The following table is the minimum prose correction before Part 2 begins. This does not require deleting the current report; add a prominent retrospective note or `REPORT_v2_ERRATA.md` so future agents do not inherit claims that the code does not support.

| Current-style claim | Revised status | Publication-safe wording now |
|---|---|---|
| “The disagreement with the paper's causal story was instruments all along” | Not identified | “Several alternative interventions produced different outcomes, but the paper's output-protected dynamic ablation has not yet been reproduced.” |
| “Static J-span causal dissociation is null on both models” | Provisional auxiliary null | “A selected static corpus-level J span produced no large effect at tested doses under the current tasks and aggregate-energy controls.” |
| “Frozen per-item J ablation is control-clean” | Provisional | “Prompt-selected J-aligned projection reduces factual-answer scores relative to the implemented random controls; geometry- and energy-matched controls remain.” |
| “OLMo capacity is ten times thinner than the paper” | Not identified | “The lab's fixed-threshold sparse-reconstruction summaries are much smaller for OLMo than Qwen; paper-comparable occupancy has not been computed.” |
| “Qwen is paper-range under the same harness” | Not identified | “Qwen is substantially larger than OLMo under the lab's current estimator; comparison to the paper requires the paper's occupancy and random-adjusted variance definitions.” |
| “Live per-token deletion measures computation deletion” | Exploratory | “Unprotected live J-direction deletion catastrophically degrades output, unlike a random live control. The paper's intended-output protection is missing.” |
| “Workspace leads CoT by 46 steps” | Exploratory mid-band, provisional late-band | “Answer-related J readouts often precede answer strings. A complete-sample, preregistered, foil-calibrated onset estimate is pending.” |
| “Externalization rescues frozen deletion” | Provisional mention recovery | “Extended thinking often re-expresses the target answer under the J intervention; recovery of final task performance has not yet been established.” |
| “Second seed exact replication” | Fresh-sample robustness | “The effect reappeared after changing item set, random pools, and seed. This is a useful bundled replication, not a seed-only sensitivity test.” |
| “Broadcast non-dissociation stands” | Not paper-comparable | “The lab's linear fan-out metric did not distinguish J directions from structured high-variance controls. The paper's MLP and attention broadcast assays remain untested.” |

---

## 7. Full review of `jspace_interp_part2_plan1.md`

The Part 2 plan contains many of the right experiments. Its main flaw is sequencing: it assumes Part 1's instruments are mostly closed and moves immediately toward model breadth. The code audit shows that instrumentation is the largest remaining uncertainty, including mismatches with the paper's exact protocol and definitions. The plan should be retained, but placed behind a publication assay gate.

### 7.1 Review of the plan's starting claims

The opening section currently states that:

- descriptive geometry replicates with a tenfold cross-model capacity difference;
- the paper's static-span causal dissociation is null on two open models;
- frozen per-item ablation is control-clean;
- the lead and rescue battery is novel and largely validated;
- instrument uncertainty is mostly closed.

These statements should be replaced before execution. A better starting state is:

> Part 1 produced a functioning J-lens pipeline and several provisional effects, but it did not yet reproduce the paper's output-protected live ablation or paper-defined occupancy, and its strongest novel frozen and CoT results require matched controls and confirmatory endpoints. Part 2 begins by repairing and validating those assays, then uses a model matrix to test preregistered explanations for whichever findings survive.

This wording is less triumphant and more powerful. It makes a successful correction publishable rather than embarrassing.

### 7.2 Hypothesis ladder review

The existing H1 to H5 ladder is insightful but omits hypotheses that the audit makes unavoidable.

#### Existing H1: Externalization

**Keep, but refine.** “Think-trained models put the workspace in tokens” is too broad. Separate:

- **H1a, learned externalization policy:** post-training changes where intermediate state is stored or how long it remains internal;
- **H1b, inference-mode externalization:** enabling visible reasoning in the same checkpoint changes workspace demand during the current run;
- **H1c, extra-compute effect:** longer autoregressive generation improves performance without requiring semantically meaningful externalized reasoning.

The OLMo lineage can address H1a. Qwen thinking on/off can address H1b. Matched filler and externally supplied rationale controls can address H1c.

#### Existing H2: Output occupancy

**Keep as a mechanistic hypothesis, but do not use the current live-collapse result as evidence for it.** First implement the paper's intended-output protection. Then define output occupancy with multiple measures:

1. overlap between active J token labels and clean output top-k;
2. cosine or principal-angle overlap between selected J projector and output-aligned subspace;
3. fraction of selected J directions classified as motor-like using held-out activation statistics;
4. effect of protecting clean output directions on fluency and reasoning;
5. dynamic occupancy during prefill, hidden reasoning, and final response.

The proposed scalar “emitted token rank in live readout” is useful but insufficient. It can be high because output and workspace content are semantically related, not because they are the same computational channel.

#### Existing H3: Training-lab specifics

**Replace “unfalsifiable” with a bounded residual hypothesis.** The open-model study cannot identify Anthropic's proprietary training recipe, but it can test whether observable checkpoint axes explain the difference. The OLMo base/SFT/DPO/RL ladder is especially valuable because the model cards expose several post-training stages. If none of the open axes reproduce the paper's signature, the remaining conclusion is not “Anthropic magic”; it is “not explained by the tested public model, architecture, scale, task, lens, or post-training axes.”

#### Existing H4: Scale

**Keep bounded.** Qwen's larger current metric does not yet weaken scale relative to the paper because the metric is non-commensurate. Reassess scale only after paper-defined occupancy is implemented. With the available compute, treat scale as a descriptive moderator rather than a fully adjudicated cause.

#### Existing H5: Instruments

**Promote to H0 and run first.** The correct question is:

- **H0a:** the paper-faithful assay reproduces its advertised positive controls on released data and a supported open model;
- **H0b:** the lab's current static and frozen results survive rank-safe, phase-resolved, paired analyses;
- **H0c:** paper-comparable occupancy and broadcast agree or disagree with the lab's exploratory proxies.

Until H0 passes, every model comparison is partly a comparison of unknown measurement behavior.

#### Add H6: Task demand and shortcut structure

The current tasks may not demand a workspace, and the current one-hop controls are not difficulty matched. H6 predicts that intervention effects emerge as latent-state load increases and that the effect is not explained by baseline difficulty, answer probability, token length, or template family.

#### Add H7: Lens-model mismatch

A corpus-averaged Jacobian may be a poor local approximation on some checkpoints, modes, prompts, or layers. H7 predicts that local Jacobians, future-only Jacobians, or better corpus fits improve readout and causal specificity. This should be tested on a tractable subset rather than assumed away.

#### Add H8: Sparse-frame geometry artifact

Prompt-selected J interventions may work because the J dictionary has a distinctive coherence and extreme-value structure, not because selected token labels identify a privileged content channel. Rotated-J and singular-spectrum-matched controls directly test this.

### 7.3 Workstream A: model matrix

#### A0: Lens-transfer gate

**Verdict: keep as a scientific transfer experiment, reject as a canonical-fit shortcut.**

A transferred lens can reveal whether Jacobian geometry is conserved across post-training. That is interesting. It cannot eliminate recipient fitting merely because probe ranks are within 15%. Causal interventions depend on direction geometry, not only readout hit rate.

Revised A0:

1. Fit a small recipient lens, for example `n=25`, before any transfer conclusion.
2. Compare donor and recipient lens on held-out readout, dictionary CKA, token-direction cosine, projector principal angles, occupancy, and a tiny causal calibration set.
3. If transfer is equivalent within preregistered margins, it may be used for exploratory broad sweeps.
4. Confirmatory headline cells still use a recipient-fitted canonical lens, preferably `n=250` or the empirically validated saturation point.

#### A1: OLMo Instruct

**Verdict: highest-value model comparison, but pin a true lineage and expand it.**

Use a matched base and post-training ladder where available:

- base checkpoint;
- Think SFT;
- Think DPO;
- final Think;
- Instruct SFT;
- Instruct DPO;
- final Instruct.

The official OLMo cards indicate common base lineage but distinct post-training branches. This is much stronger than comparing only two endpoints, because it can identify when a phenomenon appears, disappears, or rotates.

Do not call Think versus Instruct “differing only in post-training.” They share a base lineage, but the post-training datasets, objectives, and potentially checkpoint vintages differ. It is a matched-lineage natural experiment, not a randomized one-variable intervention.

Also decide whether Part 2 upgrades the original `Olmo-3-32B-Think` endpoint to `Olmo-3.1-32B-Think`. The clean primary matrix should not mix 3.0 Think with 3.1 Instruct unless the checkpoint relationship is explicitly modeled. Preferred primary pair:

- `allenai/Olmo-3.1-32B-Think`;
- `allenai/Olmo-3.1-32B-Instruct`;
- their documented shared base lineage;
- optional intermediate SFT/DPO checkpoints.

Retain the original OLMo-3-32B-Think as a historical replication cell, not the sole member of the new confirmatory pair.

#### A2: Qwen non-think

**Verdict: split into within-checkpoint mode and lineage questions.**

For Qwen3.6-27B, thinking and non-thinking are official modes of the same checkpoint. This is excellent for H1b because weights, lens, and architecture can be held fixed. It does not test post-training differences.

Run:

- official chat rendering with thinking enabled;
- official chat rendering with thinking disabled;
- greedy or deterministic teacher-forced primary readout cells;
- mode-recommended sampling only as a secondary behavioral robustness analysis;
- identical prompt content, answer rubric, token budget, and lens.

Then, separately, identify a base or prior non-hybrid Qwen checkpoint for an H1a lineage comparison. Do not use raw completion prompting as a proxy for official non-thinking mode.

#### A3: Gemma 4

**Verdict: good third family after OLMo and assay validation.**

Gemma 4 31B adds hybrid local/global attention and multimodal machinery, and it supports configurable thinking behavior. That makes it scientifically valuable, but also increases adapter risk.

The adaptation gate should cover more than final norm and logit softcapping:

1. confirm the exact residual stream hooked by `jlens`;
2. verify pre-norm/post-norm placement and final unembedding path;
3. bypass or hold the vision tower constant for text-only experiments;
4. verify local/global attention cache behavior under hooks;
5. validate that generation and teacher-forced passes use the same text path;
6. compare analytical or finite-difference directional derivatives on a tiny model slice;
7. test thinking on/off in the same weights;
8. fit a model-specific lens before any substantive comparison.

Gemma should not be allowed to consume the budget before the OLMo primary pair has complete confirmatory cells.

### 7.4 Workstream B: instrument robustness

#### B1: fitting-set scaling

**Verdict: essential, but revise the design and thresholds.**

Nested `120/250/500` fits are useful, but one curve cannot estimate corpus sampling variance. Include independent draws and a held-out quality criterion. The proposed median cosine `>0.9`, Jaccard `>=0.7`, and effect movement `<0.5` nats are arbitrary until tied to measurement error and scientific relevance.

Use equivalence margins derived from:

- between-seed variation;
- control-condition variation;
- the smallest effect that would change a headline conclusion;
- simulation-based power.

Direction-wise cosine should be supplemented by geometry and projector metrics. A per-token direction can rotate while the sparse frame's functional geometry remains stable, and vice versa.

#### B2: Dolma corpus lens

**Verdict: keep, broaden slightly.**

Use at least one pretraining-matched corpus and one generic web-text corpus, with held-out evaluation on both. This turns corpus choice into a measured domain-shift matrix rather than a single WikiText-versus-Dolma anecdote.

Also fit a task-skewed lens only as a negative or stress test. If a task-specific corpus dramatically improves the result, that indicates lens conditioning rather than a universal workspace and must be reported as such.

#### B3: frozen-logit control

**Verdict: high priority, but not sufficient alone.**

This is a strong and cheap control. Add:

- tuned-lens directions if a valid tuned lens is available;
- layer-shuffled J directions;
- J-rotated directions;
- label-shuffled J directions;
- local-gradient or per-prompt Jacobian directions on a small subset.

The scientific question is not only “J versus logit.” It is whether the mean-Jacobian transport contributes specificity beyond output alignment, dictionary geometry, and prompt-local causal gradients.

#### B4: higher dose and selection escape hatches

**Verdict: keep, but do not treat larger `k` as a free robustness axis.**

At `k=80` or `160` per layer, effective total intervention can be large and coherence may degrade. Every dose must be:

- nested within seed;
- effective-rank reported;
- energy and spectrum matched;
- calibrated for fluency and top-1 agreement;
- analyzed with a dose-by-task interaction;
- stopped if the intervention crosses a predefined general-damage boundary.

Persistence-selected spans are worth testing, but persistence must be defined on a held-out corpus to avoid selecting from the evaluation tasks.

#### B5: decoding robustness

**Verdict: secondary, not a replacement for replication.**

For causal internal mechanisms, deterministic teacher-forced log probabilities and greedy accuracy should be primary. Sampling introduces useful ecological validity but also adds variance and mode-specific recommended settings. Run multiple sampling seeds and analyze item-level success probabilities. A single temperature-0.7 replicate is not enough to characterize decoding robustness.

### 7.5 Workstream C: task battery

#### C1: parametric working-set battery

**Verdict: essential and potentially the paper's strongest open-model test.**

The examples in the current plan are good starts, but task generation must prevent shortcuts. Each family should vary:

- load;
- number of operations;
- delay length;
- distractor similarity;
- answer token length;
- baseline difficulty;
- surface template;
- relation family;
- whether the intermediate must be maintained, retrieved, transformed, or rebound.

Use automatically generated counterfactual twins where one latent value changes but surface statistics remain matched. Include a solver that verifies every generated item and a contamination audit for common facts.

Most importantly, separate **load** from **difficulty**. Fit baseline item-response curves, then compare intervention effects among items matched for baseline log probability or success. Otherwise a larger high-load drop can be a floor effect.

#### C2: paper task sets verbatim

**Verdict: mandatory and broader than currently listed.**

Run not only capacity and directed modulation. The released experiment descriptions include:

- probe-swap;
- verbal introspection;
- verbal report;
- directed modulation;
- top-down summoning;
- flexible generalization;
- language and line-count selectivity;
- ignition;
- capacity;
- dual-task competition.

Not every task needs to be a headline cell, but the exact paper multihop and selectivity tasks are necessary for causal replication, and capacity/dual-task are especially relevant to the working-memory thesis.

Adaptation rules must be frozen before outcomes are inspected. When tokenization or model capability makes verbatim use impossible, record a deterministic transformation and keep both original and adapted versions.

#### C3: hard one-hop set

**Verdict: mandatory.**

Difficulty match should use more than baseline log probability. Match or model:

- answer-token count;
- frequency;
- entity familiarity;
- prompt length;
- answer entropy;
- baseline accuracy;
- relation family;
- whether the first answer token uniquely identifies the answer.

A stronger design pairs each two-hop item with a one-hop item that asks directly for its intermediate or final fact using the same entities and answer. This reduces content and familiarity differences.

### 7.6 Workstream D: occupancy index

**Verdict: promising after protocol repair, but rename and expand.**

Call the proposed metric **output-alignment index**, not occupancy, because the paper already uses occupancy for sparse capacity. Use a family of measures rather than one scalar:

- token-label overlap with clean output top-k;
- selected-projector overlap with output-gradient subspace;
- motor-feature classification rate;
- intervention damage before and after output protection;
- overlap at prefill, reasoning, and final-answer phases;
- emitted-token rank relative to matched semantic foils;
- output-alignment change across Think/Instruct checkpoints and Qwen modes.

The preregistered prediction can remain: models or modes with stronger output alignment should suffer more fluency damage from unprotected live ablation. But the paper-faithful protected intervention is the primary causal test.

### 7.7 Workstream E: secondary follow-ups

- **Rescue disambiguation:** promote to the main CoT design, not a late optional detail.
- **`famously` attractor:** keep exploratory and low priority.
- **Eval-awareness:** defer until the main assay is complete.
- **Upstream numeric fixes:** do immediately after tests are written; do not wait until the end. The reference repository states it is not maintained and is not accepting contributions, so keep a pinned local fork or patch series even if an upstream PR is not possible.

### 7.8 Statistics section

The current plan's `n >= 60`, two seeds, bootstrap intervals, and BH-FDR are not enough by themselves.

Problems:

- `n=60` has no power rationale;
- task items are clustered by template, relation, schema, and fact family;
- two seeds do not characterize lens-fit and control-draw variance;
- FDR across a large model-by-instrument grid can obscure the confirmatory question;
- null conclusions need equivalence margins;
- behavioral and log-probability endpoints have different distributions;
- exploratory selection decisions have already been informed by Part 1.

Replace with the statistical plan in Section 12.

### 7.9 Priority order

The existing order begins `A0 -> B3 -> A1`. Replace it with:

`R0 provenance freeze -> R1 exact paper protocol -> R2 paper capacity -> R3 rank-safe paired frozen assay -> R4 task calibration -> preregistration -> OLMo lineage -> Qwen mode -> Gemma -> robustness breadth -> secondary explorations`.

The rule “if a positive appears, characterize it before adding models” is excellent and should remain.

---

## 8. Revised hypothesis table and directional predictions

The following table should be copied into `preregistration.md` before confirmatory item generation is revealed.

| Hypothesis | Primary discriminator | Prediction if supported | Prediction if weakened |
|---|---|---|---|
| H0a, protocol fidelity | Output-protected dynamic ablation on released multihop/selectivity tasks | Positive control content is removed, general text remains substantially intact, reasoning-selective effect can be estimated | Fluency still collapses or positive control fails |
| H0b, metric fidelity | Paper-defined occupancy and excess variance | Synthetic tests recover truth; open-model values are stable across solvers and fits | Values depend strongly on solver, random dictionary, or fit draw |
| H1a, learned externalization | OLMo base/SFT/DPO/final Think vs Instruct lineage | Workspace sensitivity changes systematically along Think branch or diverges between final branches | No branch-linked pattern after task and lens matching |
| H1b, inference-mode externalization | Qwen3.6 same weights, thinking on/off | Internal workspace load or intervention sensitivity differs by mode | Mode changes surface tokens but not pre-response workspace measures |
| H1c, extra compute | Real rationale vs irrelevant/filler/shuffled controls | Semantically valid reasoning yields selective rescue beyond equal compute | All long-generation arms rescue similarly |
| H2, output alignment | Output-alignment index plus protected/unprotected live ablation | Unprotected fluency damage tracks output alignment; protection restores interpretability | Damage remains unrelated to output alignment or protection |
| H4, scale/capacity | Paper-defined occupancy across models | Capacity predicts load sensitivity or dual-task interference after controls | Capacity varies without functional relationship |
| H6, workspace demand | Parametric load-by-intervention interaction | Effect increases with load at matched baseline difficulty | No interaction or effect tracks only difficulty |
| H7, mean-J mismatch | Mean J vs local/future-only J subset | Better local approximation improves readout and causal specificity | All lens variants give equivalent results |
| H8, geometry artifact | J vs J-rotated/spectrum-matched/shuffled controls | Semantic J selection exceeds geometry-preserving controls | Rotated or shuffled controls reproduce the effect |

Predefine the primary decision criterion for each hypothesis and cap the number of headline tests. Everything else is exploratory or robustness evidence.

---

## 9. Revised Part 2 execution plan

### Stage 0: Freeze, inventory, and preregister the repair campaign

**Goal:** turn the merged exploratory project into a versioned research program without corrupting prior artifacts.

#### 0.1 Create immutable study roots

```text
interpretability/jspaces/phases/phase2/
  protocol/
  adapters/
  metrics/
  experiments/
  tests/
  reports/
  preregistration/

/content/drive/MyDrive/interpret/special-lab-1/
  part2_<date>/
    exploratory/
    confirmatory/
    manifests/
    lenses/
    raw/
    derived/
    figures/
    logs/
```

Never overwrite v1 or v2 artifacts. Every Part 2 phase reads them by hash and writes new outputs.

#### 0.2 Build an artifact inventory

For every referenced v1/v2 artifact, record:

- path;
- size;
- SHA-256;
- producing script and code commit;
- model/lens/prompt dependencies;
- whether it is complete, partial, selected, or regenerated;
- whether raw per-item data exists;
- whether it is suitable for confirmatory reuse.

Mark unavailable raw data explicitly. Do not synthesize missing per-item results from aggregates.

#### 0.3 Write two preregistrations

1. **Repair preregistration:** defines validation tests for occupancy, output-protected ablation, frozen controls, phase hooks, and paired scoring.
2. **Scientific preregistration:** written only after repair passes, defines model/task hypotheses, primary endpoints, exclusions, sample sizes, and decision rules.

This separation prevents the necessary debugging process from contaminating the later confirmatory claims.

#### 0.4 Freeze item pools before labels are exposed

Generate a large task bank, hash it, and partition by family into:

- development/calibration;
- preregistered pilot;
- confirmatory holdout;
- independent replication holdout.

The coding agent may inspect development labels. It must not inspect confirmatory outcomes until the pipeline and analysis script are frozen.

### Stage 1: Assay repair and conformance tests

**This is the new highest-priority workstream. No 32B model matrix begins until it passes.**

#### 1.1 Model-adapter interface

Implement a common adapter with explicit methods for:

```python
class JSpaceModelAdapter(Protocol):
    model_id: str
    revision: str
    n_layers: int
    d_model: int

    def encode(self, text: str, *, max_length: int) -> torch.Tensor: ...
    def render(self, messages, *, thinking: bool | None) -> str: ...
    def residual_layers(self) -> Sequence[torch.nn.Module]: ...
    def final_norm_direction(self, token_ids: torch.Tensor) -> torch.Tensor: ...
    def forward_with_cache(self, ...): ...
    def clean_output_logits(self, ...): ...
    def phase_context(self, phase: Literal["prefill", "decode"]): ...
```

Each model adapter gets unit tests and a tiny end-to-end conformance run.

#### 1.2 Exact paper-faithful dynamic ablation

Implement a clean reference path before optimizing:

1. run a clean forward pass;
2. cache top-ten clean output token IDs at each scored token position;
3. compute active J directions using the paper-consistent sparse/non-negative score;
4. exclude protected output token directions;
5. build a numerical-rank-safe projector for the remaining top ten;
6. rerun with the projector at the specified layer band and token position;
7. log all selections, protected IDs, singular values, energy, and output changes.

Validate that the protected implementation no longer directly removes the intended output direction in a synthetic construction.

#### 1.3 Phase-resolved hook validation

Use sentinel activations and tiny generated sequences to prove that:

- prefill-only hooks never fire on decode tokens;
- decode-only hooks do not alter cached prompt keys/values;
- both-mode matches the old behavior where intended;
- no hook remains attached after exceptions;
- tuple versus tensor module outputs are preserved;
- cache and no-cache execution agree within tolerance.

#### 1.4 Paper-defined occupancy and variance

Implement and test:

- non-negative sparse solver;
- random-control dictionaries;
- marginal-gain crossing;
- globally centered, mergeable moments;
- excess variance at median occupancy;
- uncertainty across random controls and prompts.

Recompute OLMo and Qwen exploratory capacity values with the final algorithm before making any further capacity claim.

#### 1.5 Rank-safe projector construction

Replace raw QR everywhere with SVD or rank-revealing QR. Effective intervention rank, not requested `k`, is the scientifically meaningful dose.

#### 1.6 Full-sequence scoring

Implement conditional log probability for the complete answer token sequence, with optional length normalization and aliases. Keep first-token probability only as a diagnostic.

#### 1.7 Paired analysis pipeline

Every experiment emits per-item records. A locked analysis script computes paired differences, clustered uncertainty, equivalence tests, and multiplicity-adjusted confirmatory results.

#### 1.8 Tiny-model golden tests

Run the full pipeline on a small supported decoder. The goal is not to reproduce global workspace behavior. It is to prove:

- the lens fits and reloads;
- manifests reproduce exact hashes;
- phase hooks work;
- protected tokens are not removed;
- nested doses are nested;
- per-item outputs regenerate aggregate tables;
- interrupted descriptive runs merge to the same covariance as uninterrupted runs;
- report figures are pure functions of saved raw records.

### Stage 2: Re-audit Part 1 under the repaired assay

Use the existing OLMo-3-32B-Think lens and checkpoints where valid, but rerun decisive cells.

#### 2.1 Dynamic replication grid

On development and pilot sets:

- baseline;
- paper-faithful protected dynamic J ablation;
- unprotected dynamic J ablation, diagnostic only;
- matched dynamic random;
- matched dynamic J-rotated;
- paper-relevant layer bands and light/medium/heavy ranges;
- exact paper multihop and selectivity tasks;
- pretraining top-1 agreement or NLL as the general-processing guard.

This decides whether the asterisk collapse was primarily the missing protection and whether a reasoning-selective effect appears once the protocol is faithful.

#### 2.2 Capacity re-estimation

Recompute OLMo and Qwen using identical final solver code. Report old versus new values in an errata figure. Do not hide changes.

#### 2.3 Frozen intervention repair

Run the full control family on a small development set, then choose a preregistered confirmatory subset of controls based on which alternatives they distinguish, not which gives the prettiest contrast.

#### 2.4 CoT endpoint repair

Save every trace, reuse immutable projectors, enforce response segmentation, and rerun a small balanced direct-versus-think factorial. This determines whether a full expensive rescue campaign is justified.

#### Stage 2 decision gate

- If paper-faithful dynamic ablation produces a selective effect, Part 2's center becomes **characterization and post-training localization of a positive replication**.
- If it remains null with strong positive controls and equivalence evidence, Part 2's center becomes **a rigorous boundary-of-generalization study**.
- If positive controls fail or solver/lens instability dominates, Part 2 becomes **a methods paper about J-lens reliability**, and model breadth pauses.

### Stage 3: Build and validate the task battery

#### 3.1 Exact released tasks

Implement the released prompt sets first, with model-specific tokenization audits and baseline capability gates.

#### 3.2 Parametric latent-state tasks

Recommended families:

1. **Entity binding:** assign attributes to multiple entities, intervene after distractors, query one binding.
2. **Variable updates:** sequential assignments with overwrites and delayed query.
3. **Relational composition:** two-, three-, and four-hop chains with counterfactual twins.
4. **Stack and queue state:** push/pop or enqueue/dequeue operations described in text.
5. **Order constraints:** maintain a partial ordering and answer a relation query.
6. **Deferred arithmetic:** compute intermediates separated by unrelated text.
7. **Dual covert task:** maintain a concept while performing arithmetic or copying text.
8. **Schema planning:** generated SQL schemas with independent names, joins, distractor keys, and query goals.
9. **Program execution:** tiny deterministic code snippets with variable state and branching.
10. **List capacity:** paper-style related and unrelated item lists.

For each family, generate at least dozens of independent templates and hundreds of item instances. The unit of generalization is the template/family, not merely the rendered item.

#### 3.3 Behavioral calibration

Before intervention:

- remove items with ambiguous gold labels;
- require a prespecified baseline success band, such as 60% to 95%, for accuracy endpoints;
- retain harder items for log-probability analysis where appropriate;
- match one-hop and multihop subsets by baseline score and tokenization;
- verify that load manipulations change latent demand, not only prompt length;
- run shortcut probes and counterfactual twins.

### Stage 4: OLMo lineage, the primary confirmatory study

This is the most scientifically valuable use of GPU budget after assay repair.

#### 4.1 Checkpoints

Preferred matrix, subject to exact available revisions:

- shared 32B base;
- Think SFT;
- Think DPO;
- final Think;
- Instruct SFT;
- Instruct DPO;
- final Instruct.

At minimum, run base, final Think, and final Instruct. Intermediate stages add localization power.

#### 4.2 Canonical lenses

Fit model-specific lenses using the validated canonical corpus size and method. Also run donor-to-recipient transfer as a secondary geometry analysis.

#### 4.3 Primary endpoints

1. protected dynamic J ablation by task demand;
2. full-sequence answer log-probability and accuracy;
3. paper-defined occupancy and excess variance;
4. dual-task interference;
5. output-alignment index;
6. frozen J versus strongest geometry-matched control;
7. load-by-ablation interaction.

#### 4.4 Primary causal contrast

Do not simply compare whether one checkpoint is “significant” and another is not. Estimate the checkpoint-by-intervention interaction on matched items:

`Delta_Think - Delta_Instruct`, with the base as an anchor.

Use a hierarchical model or paired bootstrap that respects shared items and task families.

#### 4.5 Interpretive outcomes

- **Think weaker than Instruct, with base/internals supporting a branch trend:** supports learned externalization.
- **Think and Instruct both positive:** workspace signature generalizes across post-training.
- **Both null with assay positive controls:** weakens post-training explanation in this lineage.
- **Base differs from both post-trained models:** supports a general assistant/post-training transformation rather than Think-specific externalization.
- **Only frozen content effect changes:** suggests retrieval/channel geometry changes without a global-workspace dissociation.

### Stage 5: Qwen within-checkpoint mode study

Use the same Qwen3.6-27B weights and model-specific canonical lens.

#### 5.1 Conditions

- official thinking enabled;
- official thinking disabled;
- identical user prompts and fixed system prompt;
- deterministic primary teacher-forced metrics;
- behavioral sampling under each mode's recommended settings as secondary.

#### 5.2 Questions

- Does paper-defined occupancy change before response generation by requested mode?
- Does output alignment change?
- Does protected dynamic ablation have a different task-selective effect?
- Does visible reasoning selectively rescue intervention damage beyond equal-compute controls?
- Does the same immutable projector behave differently when the generation policy externalizes reasoning?

This is a clean inference-mode study, not a post-training study.

### Stage 6: Gemma 4 architecture study

Run only after OLMo primary cells and Qwen mode cells are banked.

#### 6.1 Adaptation gate

Pass model-wrapper derivative tests, lens sanity, text-only path validation, thinking-mode rendering, cache/hook conformance, and numerical-memory benchmarks.

#### 6.2 Minimal decisive battery

- paper-defined occupancy;
- exact paper multihop/selectivity positive controls;
- protected dynamic ablation;
- hard one-hop versus multihop;
- output alignment;
- thinking on/off within checkpoint;
- strongest frozen geometry control.

Do not begin a full 50-cell grid until the minimal battery yields interpretable baselines.

### Stage 7: Mechanistic localization and rescue

Only after a robust positive or bounded null exists.

#### 7.1 Layer and time localization

Use nested bands, single-layer sweeps, prefill/decode factorial, and per-step removed-energy logs. Correct for searching across layers.

#### 7.2 Content specificity

Track known intermediates, wrong intermediates, answer tokens, and matched unrelated concepts. Test whether intervention selectively changes the intended computation rather than general confidence.

#### 7.3 Rescue and re-entry

Measure whether deleted content:

- re-enters the J-space;
- is re-retrieved from context;
- is re-derived from another fact path;
- is copied from supplied rationale;
- appears in reasoning but fails to affect the final answer.

Use causal mediation language only if the required assumptions are defended. Otherwise report temporal and intervention interactions.

### Stage 8: Confirmation, independent rerun, and release

#### 8.1 Freeze analysis

Tag the code, container, task bank, lens artifacts, and analysis plan before opening the confirmatory holdout.

#### 8.2 Confirmatory run

Run only preregistered primary cells. Do not add conditions mid-run. Any surprise follow-up is labeled exploratory and uses a separate partition.

#### 8.3 Independent rerun

A second operator or fresh agent should reproduce the primary table from a fresh machine using only the public instructions and released artifacts. At least one canonical lens should be refit from source rather than copied.

#### 8.4 Release package

Release:

- code and tests;
- environment lock/container digest;
- exact model revisions;
- prompt generators and frozen hashes;
- all nonrestricted prompts;
- raw per-item metrics;
- aggregate tables and figures;
- lens metadata and, where licensing/storage permits, lens checkpoints;
- a one-command or staged reproduction guide;
- negative and failed runs;
- preregistrations and deviations;
- errata for Part 1.

---

## 10. Concrete implementation patterns

The snippets below are deliberately small and testable. They are design patterns, not blind drop-in replacements for every model wrapper.

### 10.1 Numerical-rank-safe projector

Raw QR returns the requested number of columns even when selected atoms are nearly dependent. Use SVD, record the singular spectrum, and define dose by effective rank.

```python
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BasisResult:
    basis: torch.Tensor          # [d_model, effective_rank]
    singular_values: torch.Tensor
    effective_rank: int
    requested_rows: int
    condition_number: float | None


def orthonormal_basis_from_rows(
    rows: torch.Tensor,
    *,
    relative_tolerance: float = 1e-5,
    absolute_tolerance: float = 1e-7,
) -> BasisResult:
    """Return the numerical column span of row vectors.

    Args:
        rows: Tensor shaped [n_rows, d_model].
        relative_tolerance: Keep s_i when s_i > relative_tolerance * s_max.
        absolute_tolerance: Absolute singular-value floor.
    """
    if rows.ndim != 2:
        raise ValueError(f"expected [n_rows, d_model], got {tuple(rows.shape)}")
    if rows.shape[0] == 0:
        empty = rows.new_zeros((rows.shape[1], 0), dtype=torch.float32)
        return BasisResult(empty, rows.new_zeros(0), 0, 0, None)

    matrix = rows.float().T.contiguous()  # [d_model, n_rows]
    u, s, _ = torch.linalg.svd(matrix, full_matrices=False)
    if s.numel() == 0 or not torch.isfinite(s).all():
        raise FloatingPointError("non-finite singular spectrum")

    threshold = max(absolute_tolerance, relative_tolerance * float(s[0]))
    rank = int((s > threshold).sum().item())
    basis = u[:, :rank].contiguous()
    condition = None if rank == 0 else float(s[0] / s[rank - 1])

    # Testable invariant: Q^T Q = I on retained columns.
    if rank:
        eye = torch.eye(rank, device=basis.device, dtype=basis.dtype)
        if not torch.allclose(basis.T @ basis, eye, atol=2e-5, rtol=2e-5):
            raise AssertionError("basis is not orthonormal")

    return BasisResult(
        basis=basis,
        singular_values=s.detach().cpu(),
        effective_rank=rank,
        requested_rows=rows.shape[0],
        condition_number=condition,
    )
```

Every causal table should report requested `k` and achieved effective rank. Controls should match effective rank, not merely requested row count.

### 10.2 Output-protected dynamic selection

The exact activation score must follow the validated paper replication path. The central safeguard is simple: exclude token labels that the clean run is likely to output.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicSelection:
    selected_ids: torch.Tensor
    protected_ids: torch.Tensor
    raw_scores: torch.Tensor
    basis_result: BasisResult


def select_output_protected_j_basis(
    activation: torch.Tensor,
    dictionary: torch.Tensor,
    clean_logits: torch.Tensor,
    *,
    k: int = 10,
    protect_top_k: int = 10,
    nonnegative: bool = True,
) -> DynamicSelection:
    """Select active J rows while protecting intended output token labels."""
    h = activation.float().reshape(-1)
    d = torch.nn.functional.normalize(dictionary.float(), dim=1)
    scores = d @ h

    protected = clean_logits.float().topk(protect_top_k).indices
    masked = scores.clone()
    masked[protected] = -torch.inf
    if nonnegative:
        masked[scores <= 0.0] = -torch.inf

    finite = torch.isfinite(masked)
    available = int(finite.sum().item())
    take = min(k, available)
    selected = masked.topk(take).indices
    basis = orthonormal_basis_from_rows(d[selected])

    return DynamicSelection(
        selected_ids=selected.detach().cpu(),
        protected_ids=protected.detach().cpu(),
        raw_scores=scores[selected].detach().cpu(),
        basis_result=basis,
    )
```

Required tests:

- no protected ID appears in `selected_ids`;
- a synthetic “intended output” direction remains untouched;
- requested `k` can shrink when too few valid positive directions exist;
- duplicate dictionary rows do not inflate effective rank;
- float16 and float32 selection agree above a defined tolerance;
- selection is deterministic under fixed inputs.

### 10.3 Explicit prefill/decode phase control

Do not infer phase from prose. Make it part of the execution state and log it.

```python
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

Phase = Literal["inactive", "prefill", "decode"]
_CURRENT_PHASE: ContextVar[Phase] = ContextVar("jspace_phase", default="inactive")


@contextmanager
def intervention_phase(phase: Phase):
    token = _CURRENT_PHASE.set(phase)
    try:
        yield
    finally:
        _CURRENT_PHASE.reset(token)


class PhaseControlledAblator:
    def __init__(self, layers, projectors, active_phases: set[str]):
        self.layers = layers
        self.projectors = projectors
        self.active_phases = active_phases
        self.handles = []
        self.fire_counts = {"prefill": 0, "decode": 0}

    def _hook(self, layer_index: int):
        def hook(module, inputs, output):
            phase = _CURRENT_PHASE.get()
            if phase not in self.active_phases:
                return output
            self.fire_counts[phase] += 1

            hidden = output if torch.is_tensor(output) else output[0]
            q = self.projectors[layer_index].to(hidden.device, torch.float32)
            h32 = hidden.float()
            edited = h32 - (h32 @ q) @ q.T
            edited = edited.to(hidden.dtype)
            return edited if torch.is_tensor(output) else (edited, *output[1:])

        return hook
```

The generation wrapper should execute the prompt with `intervention_phase("prefill")`, then each cached token with `intervention_phase("decode")`. Assert expected fire counts at the end of every item.

### 10.4 Mergeable global moments

For scalar centered variance summaries, persist `n`, mean, and `M2`. For full PCA, either process one layer at a time with a mergeable covariance or use a tested streaming randomized-PCA sketch.

```python
from dataclasses import dataclass


@dataclass
class RunningVectorMoments:
    n: int
    mean: torch.Tensor
    m2: torch.Tensor

    @classmethod
    def empty(cls, dimension: int, *, dtype=torch.float64):
        return cls(
            n=0,
            mean=torch.zeros(dimension, dtype=dtype),
            m2=torch.zeros((dimension, dimension), dtype=dtype),
        )

    def update(self, batch: torch.Tensor) -> None:
        x = batch.detach().to("cpu", dtype=self.mean.dtype)
        if x.ndim != 2 or x.shape[1] != self.mean.numel():
            raise ValueError("batch shape mismatch")
        m = x.shape[0]
        if m == 0:
            return

        batch_mean = x.mean(dim=0)
        centered = x - batch_mean
        batch_m2 = centered.T @ centered

        if self.n == 0:
            self.n = m
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
            return

        total = self.n + m
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + torch.outer(delta, delta) * (self.n * m / total)
        self.mean += delta * (m / total)
        self.n = total

    def merge(self, other: "RunningVectorMoments") -> None:
        if other.n == 0:
            return
        if self.n == 0:
            self.n = other.n
            self.mean.copy_(other.mean)
            self.m2.copy_(other.m2)
            return

        total = self.n + other.n
        delta = other.mean - self.mean
        self.m2 += other.m2 + torch.outer(delta, delta) * (
            self.n * other.n / total
        )
        self.mean += delta * (other.n / total)
        self.n = total

    def covariance(self, *, unbiased: bool = True) -> torch.Tensor:
        denom = self.n - 1 if unbiased else self.n
        if denom <= 0:
            raise ValueError("not enough samples")
        return self.m2 / denom
```

Golden test: randomly partition the same activation matrix into many batchings, interrupt and resume, merge in different orders, and verify the covariance matches a direct calculation within tolerance.

### 10.5 Full answer-sequence conditional log probability

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class SequenceScore:
    token_ids: list[int]
    token_logprobs: list[float]
    sum_logprob: float
    mean_logprob: float


@torch.no_grad()
def conditional_sequence_logprob(
    hf_model,
    prompt_ids: torch.Tensor,
    answer_ids: torch.Tensor,
) -> SequenceScore:
    """Score P(answer | prompt) over the full answer token sequence."""
    if prompt_ids.ndim != 2 or answer_ids.ndim != 2:
        raise ValueError("expected batched [1, seq] tensors")
    if prompt_ids.shape[0] != 1 or answer_ids.shape[0] != 1:
        raise ValueError("this helper expects batch size one")
    if answer_ids.shape[1] == 0:
        raise ValueError("answer must contain at least one token")

    full = torch.cat([prompt_ids, answer_ids], dim=1)
    logits = hf_model(input_ids=full, use_cache=False).logits
    start = prompt_ids.shape[1] - 1
    stop = full.shape[1] - 1
    prediction_logits = logits[:, start:stop, :]
    log_probs = prediction_logits.log_softmax(dim=-1)
    gathered = log_probs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)
    values = gathered[0].float().cpu()

    return SequenceScore(
        token_ids=answer_ids[0].tolist(),
        token_logprobs=values.tolist(),
        sum_logprob=float(values.sum()),
        mean_logprob=float(values.mean()),
    )
```

For aliases, score each prespecified valid answer form and define the aggregation rule before results are inspected. Do not choose the best ad hoc string after generation.

### 10.6 Paired cluster bootstrap

Compute item-level intervention deltas first, then resample independent clusters such as template or fact family.

```python
from collections.abc import Callable

import numpy as np
import pandas as pd


def paired_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    cluster_column: str,
    item_column: str,
    condition_column: str,
    score_column: str,
    treatment: str,
    baseline: str,
    statistic: Callable[[np.ndarray], float] = np.mean,
    draws: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    pivot = frame.pivot_table(
        index=[cluster_column, item_column],
        columns=condition_column,
        values=score_column,
        aggfunc="first",
    ).dropna(subset=[treatment, baseline])
    pivot["delta"] = pivot[treatment] - pivot[baseline]

    clusters = pivot.index.get_level_values(cluster_column).unique().to_numpy()
    if len(clusters) < 2:
        raise ValueError("need at least two independent clusters")

    by_cluster = {
        cluster: pivot.xs(cluster, level=cluster_column)["delta"].to_numpy()
        for cluster in clusters
    }
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        values = np.concatenate([by_cluster[c] for c in selected])
        samples[draw] = statistic(values)

    observed = statistic(pivot["delta"].to_numpy())
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "estimate": float(observed),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_items": int(len(pivot)),
        "n_clusters": int(len(clusters)),
    }
```

For a multi-model confirmatory study, use a mixed model as the primary or complementary analysis, with item family and model/lens fit represented explicitly. The bootstrap remains a transparent robustness analysis.

### 10.7 Bootstrap equivalence decision

A bounded null requires a smallest effect size of interest, `delta`. A simple preregistered rule is that the entire two-sided confidence interval lies inside `[-delta, +delta]`.

```python
def equivalence_from_interval(
    estimate: float,
    ci_low: float,
    ci_high: float,
    *,
    smallest_effect: float,
) -> bool:
    if smallest_effect <= 0:
        raise ValueError("smallest_effect must be positive")
    return ci_low > -smallest_effect and ci_high < smallest_effect
```

For formal TOST, use a vetted statistics package and a 90% interval corresponding to two one-sided 5% tests. Report the margin and its justification prominently.

### 10.8 Nested random bases and doses

```python
def seeded_random_orthobasis(
    dimension: int,
    max_rank: int,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = torch.randn(max_rank, dimension, generator=generator)
    basis = orthonormal_basis_from_rows(rows).basis
    if basis.shape[1] < max_rank:
        raise RuntimeError("unexpected rank loss in random basis")
    return basis.to(device)

# Dose k is always random_basis[:, :k], never a newly drawn subspace.
```

### 10.9 Immutable selection artifacts

The projector used in direct-answer and thinking arms must be byte-identical.

```python
@dataclass(frozen=True)
class FrozenSelectionArtifact:
    item_id: str
    prompt_sha256: str
    lens_sha256: str
    layer_to_selected_ids: dict[int, list[int]]
    layer_to_basis_sha256: dict[int, str]
    layer_to_effective_rank: dict[int, int]
    selection_rule: str
    selection_phase: str
```

Serialize basis tensors separately, hash them, and reference them from the artifact. Never recompute a “same” projector from a differently rendered prompt.

### 10.10 Artifact hashing

```python
import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
```

Every phase should refuse to reuse a cached artifact when any declared input hash differs.

---

## 11. Experiment specification in publication-ready form

### 11.1 Confirmatory unit of analysis

The independent unit is not always a rendered prompt. Define it by task family:

- factual chains: relation/fact family;
- generated programs: generator template and program skeleton;
- SQL: schema family;
- dual-task: task-pair template;
- prose: source document;
- list capacity: sampled vocabulary block or list seed.

Items sharing a template or schema are repeated measurements, not fully independent observations.

### 11.2 Primary outcomes

Use a small set of primary outcomes:

1. **Full-answer conditional log-probability delta** under intervention versus paired baseline.
2. **Exact or rubric-graded final-answer accuracy** for behavior.
3. **General-processing preservation:** pretraining top-1 agreement, NLL delta, or KL divergence on held-out text.
4. **Paper-defined occupancy** and random-adjusted variance.
5. **Load-by-intervention interaction.**

Everything else, including first-token probability, top-eight detector hits, divergence anecdotes, and selected-token word clouds, is secondary or exploratory.

### 11.3 Positive controls

Every intervention family needs positive controls that prove the mechanism is capable of affecting the target:

- swap a known intermediate to a counterfactual value on released probe-swap tasks;
- remove a directly output-aligned answer direction without protection, diagnostic only;
- intervene on a task where the selected concept is known to control the answer;
- verify that protected tokens truly remain unablated;
- inject or swap a direction and confirm the predicted logit shift.

A null without a functioning positive control is assay failure, not evidence of absence.

### 11.4 Negative controls

Use controls matched to the alternative explanation:

| Alternative | Necessary control |
|---|---|
| removed energy | item- and phase-matched energy |
| effective rank | rank-matched basis |
| dictionary coherence | J-rotated or Gram-matched dictionary |
| output alignment | clean-output protection and logit directions |
| extreme-value candidate pool | equal-size candidate pool and label shuffle |
| semantic content | wrong-intermediate and unrelated-content J directions |
| extra compute | equal-forward-pass filler or shuffled rationale |
| prompt re-encoding | prefill/decode factorial |
| general model damage | prose NLL, top-1 agreement, grammar, and generation audits |

### 11.5 Randomization

Within each item:

- randomize condition execution order;
- clear caches or prove cache isolation;
- use common random numbers for sampled decoding where feasible;
- generate all seed-specific control bases before outcomes;
- keep doses nested;
- blind human graders to condition and model.

### 11.6 Exclusions

Predefine exclusions such as:

- baseline tokenizer cannot represent required single-token target when the protocol requires one;
- baseline model fails a released positive-control task below a set floor;
- malformed generation or infrastructure failure;
- non-finite lens or projector values;
- effective rank below a minimum, reported rather than silently padded;
- answer ambiguity found by the gold solver before unblinding.

Do not exclude items because the intervention effect is unusual or because the model gave an inconvenient but valid answer.

### 11.7 Human grading

For SQL, prose, or complex answers:

- write a deterministic parser wherever possible;
- use two blinded graders for residual cases;
- report inter-rater agreement;
- adjudicate disagreements under a frozen rubric;
- retain raw outputs and grader labels.

LLM-as-judge can be a secondary analysis, not the sole ground truth for the main causal table.

---

## 12. Statistical analysis plan

### 12.1 General principles

1. Analyze paired item-level effects.
2. Cluster by the true independent generation unit.
3. Separate confirmatory and exploratory families.
4. Report effect sizes and uncertainty, not only p-values.
5. Use equivalence tests for bounded null claims.
6. Include lens-fit, control-seed, and decoding uncertainty where they are material.
7. Do not compare “significant in A” with “not significant in B.” Test the A-versus-B interaction directly.
8. Predefine how missing, malformed, and unclosed reasoning outputs are handled.

### 12.2 Continuous outcomes

For full-sequence answer log probability, the cleanest raw outcome is the paired delta:

`delta_i = score_i(intervention) - score_i(baseline)`.

A confirmatory mixed model can use:

```text
delta ~ intervention_family * checkpoint * task_load
      + baseline_score
      + answer_token_count
      + (1 + intervention_family | item_family)
      + (1 | lens_fit)
      + (1 | control_seed)
```

Depending on sample size and convergence, simplify random slopes according to a preregistered fallback order. Do not choose the model that gives the preferred p-value.

Report:

- marginal mean delta per checkpoint, load, and task family;
- checkpoint-by-intervention and load-by-intervention contrasts;
- 95% confidence or credible intervals;
- standardized effect only as a secondary aid;
- raw-score distributions and paired scatterplots.

### 12.3 Binary outcomes

Use a binomial mixed model for exact final-answer accuracy:

```text
correct ~ condition * checkpoint * task_load
        + baseline_difficulty
        + (1 + condition | item_family)
        + (1 | lens_fit)
```

Also report paired risk differences, which are easier to interpret than odds ratios. For small cells, use exact paired methods or a Bayesian hierarchical model rather than a fragile asymptotic fit.

### 12.4 Null and equivalence claims

Before confirmatory runs, define the smallest effect that would matter scientifically.

Possible planning defaults, to be calibrated in the pilot:

- answer log probability: `0.5` nats per answer sequence;
- accuracy: `10` percentage points;
- prose NLL: `0.10` nats per token;
- pretraining top-1 agreement: `5` percentage points;
- occupancy: `20%` relative change or a model-specific absolute margin based on measurement repeatability.

These are starting points, not sacred constants. Justify final margins using:

- Part 1 observed variability;
- what would change the substantive conclusion;
- positive-control effect sizes;
- measurement repeatability across lens fits and seeds;
- power simulations.

A static null becomes publishable only when the confidence interval lies within the preregistered equivalence bounds, the assay's positive control succeeds, and general-damage controls remain within their own bounds.

### 12.5 Multiplicity

Define a small confirmatory family, for example:

1. OLMo Think versus Instruct interaction under protected dynamic ablation on high-load tasks;
2. OLMo base versus post-trained interaction;
3. J versus J-rotated frozen effect on full-answer log probability;
4. load-by-protected-ablation interaction;
5. Qwen thinking-on versus thinking-off interaction.

Use Holm correction for this small primary family. Use BH-FDR for prespecified secondary families. Treat layer sweeps, token-level searches, attractor forensics, qualitative examples, and unplanned task subsets as exploratory.

Do not apply one giant FDR correction across unrelated descriptive, causal, and engineering metrics. Define scientifically coherent families.

### 12.6 Power and sample size

Do not set every cell to `n=60` by habit. Use simulation after the repaired pilot.

For each primary contrast:

1. estimate baseline score distribution and item-family intraclass correlation;
2. estimate paired residual variance;
3. choose the smallest effect of interest;
4. simulate the planned mixed model across candidate numbers of families, items per family, lens fits, and control seeds;
5. choose a design with at least 80% or preferably 90% power for the primary effect and adequate precision for equivalence;
6. inflate for expected exclusions and failed baselines.

Favor more independent families over many near-duplicate items. Thirty templates with four items each are often more informative than three templates with forty items each.

### 12.7 Lens-fit replication

For the primary OLMo endpoints:

- minimum: two independently sampled canonical lens fits on Think and Instruct;
- preferred: three fits for the endpoint pair or one `n=500` canonical plus two smaller independent stability fits;
- fit sample should be independent of all evaluation text;
- transferred lenses are separate conditions, not substitutes.

For Qwen and Gemma, one canonical lens can support an extension if the primary claim is not lens-fit generalization, but at least one independent smaller fit or donor-versus-recipient check should quantify lens uncertainty.

### 12.8 Decoding seeds

Teacher-forced log-probability outcomes need no decoding seed and should carry much of the primary causal burden. For sampled behavioral outcomes:

- use common random numbers across paired conditions where implementation permits;
- run enough seeds to estimate per-item success probability, not a single lucky trajectory;
- model decoding seed as a repeated measurement;
- keep mode-specific sampling settings secondary to deterministic primary metrics.

### 12.9 Missing and malformed outputs

For thinking-mode behavior, define mutually exclusive outcomes:

- closed reasoning and valid final answer;
- closed reasoning and invalid final answer;
- unclosed reasoning at token cap;
- malformed chat structure;
- infrastructure failure.

Primary intent-to-treat scoring should count an unclosed run as failure on final-answer accuracy unless the preregistration specifies a deterministic extraction rule. Report closure separately. Do not switch to answer-anywhere because closure is low after seeing the result.

### 12.10 Qualitative examples

Select examples by an automated, preregistered rule, such as largest positive, median, and largest negative paired delta within each condition. Also include random examples. Do not present only narratively satisfying traces.

---

## 13. Reproducibility and software-quality requirements

### 13.1 Dependency and environment lock

Add one of:

- `uv.lock` plus pinned Python version;
- a fully hashed `requirements.txt`;
- a Dockerfile or Apptainer definition with immutable base digest.

Pin:

- PyTorch;
- Transformers;
- CUDA runtime and driver compatibility;
- `datasets`;
- `jlens` commit or vendored patch commit;
- tokenizer dependencies;
- plotting and statistics packages.

A `pip freeze` captured after the run is useful forensic data but not a reproducible build specification.

### 13.2 Model and dataset pins

Resolve each Hugging Face model and tokenizer to a commit SHA and record file hashes. Pin dataset revisions and split fingerprints. Cache manifests should prove that a resumed run loaded the same bytes.

### 13.3 Config-driven paths

Replace hard-coded Drive constants with a typed configuration file and CLI overrides:

```yaml
study_id: jspace-part2-confirmatory-v1
artifact_root: /content/drive/MyDrive/interpret/special-lab-1/part2_2026_08_xx
local_cache_root: /content/jspace_cache
model:
  id: allenai/Olmo-3.1-32B-Think
  revision: <sha>
lens:
  corpus_manifest: config/corpora/dolma3_fit_500_seed0.jsonl
  source_layers: [20, 22, 24]
  target_layer: 63
experiment:
  item_manifest: config/tasks/confirmatory_holdout.jsonl
  phases: [prefill, decode]
```

### 13.4 Test suite

Minimum unit tests:

- rank-safe basis on duplicate and near-duplicate rows;
- clean-output protection;
- phase hook firing;
- full-sequence log probability against a hand calculation;
- nested doses;
- random-basis reproducibility;
- streaming moment merge equivalence;
- sparse solver on synthetic known support;
- artifact hash invalidation;
- tokenizer alias and multi-token scoring;
- cluster bootstrap on a toy dataset;
- report aggregation from raw records.

Minimum integration tests:

- tiny model lens fit, save, load, and readout;
- one paper-style swap;
- one protected dynamic ablation;
- one frozen control family item;
- interrupted/resumed descriptive run;
- raw-to-report regeneration.

### 13.5 Continuous integration

CPU CI should run formatting, type checking, unit tests, synthetic solver tests, and report regeneration from tiny fixtures. A scheduled or manual GPU smoke test can run on a small model.

### 13.6 Logging

Use structured JSONL events with:

- timestamp;
- phase and item;
- model/lens/manifest hashes;
- condition and dose;
- GPU memory;
- selected and protected directions;
- effective rank and removed energy;
- score and raw output path;
- exceptions and retries.

Human-readable logs can be rendered from these events.

### 13.7 Data release

Raw model text can be large but should be released where licensing permits. At minimum release:

- per-item scores;
- token IDs;
- selected/protected direction IDs;
- projector statistics;
- generation metadata;
- hashes pointing to any restricted raw artifact.

A reviewer must be able to regenerate every table and confidence interval without loading a 32B model.

### 13.8 Reproduction command surface

Provide commands such as:

```bash
python -m jspace_part2.audit_environment --config configs/olmo31_think.yaml
python -m jspace_part2.fit_lens --config configs/olmo31_think.yaml
python -m jspace_part2.validate_lens --config configs/olmo31_think.yaml
python -m jspace_part2.run_experiment --experiment protected_dynamic_pilot
python -m jspace_part2.analyze --study part2_confirmatory_v1
python -m jspace_part2.render_report --study part2_confirmatory_v1
```

Each command should be idempotent, resumable, and refuse incompatible caches.

### 13.9 Independent reproduction drill

Before public release, hand only the repository, artifact bundle, and reproduction instructions to a fresh agent or collaborator. Record every undocumented assumption they encounter. Fix the instructions, then repeat from a clean environment.

---

## 14. Reviewer-objection matrix, revised

| Reviewer objection | Current status | Experiment or artifact that closes it |
|---|---|---|
| “You did not reproduce the published intervention” | Open, critical | Stage 1.2 output-protected dynamic ablation |
| “Your capacity metric is not theirs” | Open, critical | Stage 1.4 paper-defined occupancy and excess variance |
| “Static span is not the paper's live J-space” | Open, conceptual | Reframe static as auxiliary; run exact dynamic protocol |
| “Your null is merely low power” | Open | Power simulation plus equivalence bounds |
| “Your controls remove different energy” | Partly addressed globally | Per-item, per-phase energy matching |
| “Your controls have different geometry” | Open | J-rotated, spectrum-matched, rank-matched controls |
| “QR padded duplicate directions” | Open | SVD rank-safe bases and effective-rank reporting |
| “You changed prompt encoding, not working memory” | Open | prefill/decode factorial |
| “First-token scoring is misleading” | Open | full-sequence conditional log probability |
| “One-hop is easier than two-hop” | Open | paired, difficulty-matched one-hop set |
| “Your tasks do not demand a workspace” | Open | parametric load and dual-task battery |
| “Your task `n` is pseudo-replicated” | Open for SQL/templates | many independent families and clustered inference |
| “The lens is underfit” | Open | nested fit-size curves and independent draws |
| “The corpus is mismatched” | Open | WikiText/Dolma domain matrix |
| “The transferred lens is invalid” | Open | recipient fit plus transfer equivalence analysis |
| “The J effect is just logit alignment” | Open | logit, tuned, local-J, and output-protection controls |
| “The J effect is just dictionary extreme values” | Open | equal pool, label shuffle, and J-rotated controls |
| “The lead detector had many chances to fire” | Open | calibrated complete-sample target-versus-foil detector |
| “The lead sample was selected” | Open | save and analyze all traces |
| “The rescue is answer mention, not behavior” | Open | fixed post-reasoning final-answer endpoint |
| “Longer compute, not reasoning, caused rescue” | Open | equal-compute filler, shuffled rationale, right/wrong rationale |
| “Qwen was not actually in official non-thinking mode” | Open | official chat-template mode toggle |
| “Cross-model code paths differ” | Open | adapter conformance tests and common estimands |
| “Broadcast was measured differently” | Open | MLP gain, attention OV gain/label preservation, J-rotated control |
| “You overfit the analysis during the first campaign” | Open | hidden holdout and preregistered confirmatory run |
| “The result cannot be reproduced from the repo” | Open | manifests, locks, raw records, and independent rerun |
| “A null on open models says nothing about Claude” | Inherently bounded | phrase as generalization boundary, not direct Claude refutation |

---

## 15. Priority order and stop rules

### 15.1 New priority order

1. **R0:** artifact inventory, environment lock, exact source pins.
2. **R1:** paper-faithful output-protected dynamic ablation.
3. **R2:** paper-faithful occupancy and variance.
4. **R3:** rank-safe frozen intervention with geometry-matched controls.
5. **R4:** full-sequence scoring, paired clustered analysis, and phase factorial.
6. **R5:** exact paper tasks and task calibration.
7. **P0:** scientific preregistration and hidden holdout freeze.
8. **A1:** OLMo base/Think/Instruct primary matrix.
9. **C1:** load-by-ablation study within OLMo matrix.
10. **A2:** Qwen thinking-on/off within-checkpoint contrast.
11. **B1/B2:** fit-size and corpus robustness where primary conclusions require it.
12. **A3:** Gemma 4 minimal decisive battery.
13. **D:** output-alignment mechanism analysis.
14. **CoT:** confirmatory lead and rescue factorial.
15. **Broadcast:** full structural and causal assay if budget permits.
16. **Secondary:** eval-awareness, attractors, qualitative forensics.

### 15.2 Hard stop rules

- Do not add models when the paper-faithful positive control fails.
- Do not call a static effect null without equivalence evidence.
- Do not call a control matched unless energy, effective rank, and the intended matching variable are measured on the evaluated items.
- Do not reuse a lens across checkpoints for a headline cell without a recipient-fitted comparison.
- Do not promote answer-anywhere to a behavioral outcome.
- Do not inspect confirmatory holdout outcomes while changing code or thresholds.
- Do not thin every workstream into tiny cells. Finish the primary assay and OLMo matrix first.
- If a robust positive appears, pause breadth and map dose, layer, phase, task demand, and control specificity.
- If the repaired assay overturns a Part 1 headline, publish the correction plainly and regenerate the claim ledger.

### 15.3 Go/no-go gates

| Gate | Pass condition | Failure response |
|---|---|---|
| G0 Provenance | all inputs pinned and hashable | stop, repair environment |
| G1 Solver | synthetic recovery and batching/resume invariants pass | stop, fix numerics |
| G2 Lens | hidden validation and negative controls pass | refit or revise lens |
| G3 Intervention | protected IDs untouched; phase hooks correct; rank/energy logged | stop, fix intervention |
| G4 Positive control | expected swap or ablation effect appears without general damage | assay not ready |
| G5 Task | baseline capability and shortcut audits pass | regenerate/calibrate tasks |
| G6 Pilot precision | planned design can resolve SESOI/equivalence | increase families or revise endpoint |
| G7 Confirmatory freeze | code, task bank, and analysis tagged | no holdout access before pass |
| G8 Independent rerun | primary table reproduces within margins | investigate before release |

---

## 16. Compute planning

The original Part 2 estimate of roughly 45 to 55 GPU-hours is a useful lower-bound based on Part 1 rates, but it does not include assay repair, recipient-fitted confirmation lenses, repeated fits, expanded task families, or an independent rerun.

Use benchmarked planning after Stage 1. The following is a rough campaign envelope, not a promise:

| Tier | Scope | Planning range |
|---|---|---:|
| Assay repair | tiny-model tests, OLMo pilot reruns, capacity recompute, frozen controls | 8 to 15 GPU-hours plus CPU engineering |
| Minimal publishable OLMo study | repair, base/Think/Instruct, exact paper tasks, load battery, two lens fits on primary endpoints, holdout | 35 to 60 GPU-hours |
| Strong cross-model study | minimal tier plus Qwen mode contrast, Gemma minimal battery, fit/corpus robustness | 70 to 110 GPU-hours |
| Full paper package | intermediate OLMo checkpoints, full broadcast, confirmatory CoT, independent refits/rerun | 100 to 160 GPU-hours |

The ranges may fall if lenses transfer computationally or rise if fit-size and architecture adaptation are expensive. After one benchmark per model, replace ranges with measured per-phase budgets in the preregistration.

### 16.1 Budget allocation principle

Spend compute in this order:

1. assay validity;
2. independent task families;
3. primary matched-lineage contrast;
4. replication across lens fits;
5. secondary model families;
6. dense robustness grids;
7. attractive but weakly identified side stories.

One extra independent lens or task family is usually worth more than another near-duplicate dose cell.

---

## 17. Publication routes and claim templates

A publishable result does not require confirming the paper. It requires a clear estimand, a functioning assay, uncertainty that excludes important alternatives, and reproducible evidence.

### 17.1 Outcome A: positive open-model replication

Possible title direction:

> **A Verbalizable Working-Memory Channel in Open Language Models: Faithful Replication and Post-Training Localization**

Claim template:

> Under the published output-protected dynamic intervention and paper-defined occupancy metric, we reproduce a reasoning-selective J-space effect in [models]. A matched OLMo lineage shows that the effect [emerges, weakens, or shifts] during [post-training stage], while general text prediction remains within preregistered preservation bounds.

Required evidence:

- exact protocol;
- exact or adapted released tasks;
- general-processing controls;
- OLMo lineage interaction;
- at least one independent lens fit and holdout;
- geometry-matched controls;
- independent rerun.

### 17.2 Outcome B: robust boundary-of-generalization null

Possible title direction:

> **Where the J-Space Workspace Dissociation Does Not Generalize: A Preregistered Open-Model Replication Across Post-Training Regimes**

Claim template:

> The released J-lens readout and positive-control manipulations function on [models], but the published reasoning-selective ablation remains within preregistered equivalence bounds across [checkpoints/tasks/doses], while matched controls and general-processing measures validate assay sensitivity. The result bounds generalization to the tested open-model lineages and does not directly refute the Claude result.

Required evidence:

- successful positive controls;
- equivalence, not merely non-significance;
- sufficient power;
- paper-faithful intervention;
- paper-faithful metrics;
- broad task demand;
- matched-lineage and at least one additional family;
- reproducibility.

### 17.3 Outcome C: method instability

Possible title direction:

> **How Stable Is the Jacobian Lens? Corpus, Solver, Checkpoint, and Intervention Sensitivity in 30B-Scale Models**

Claim template:

> J-lens readouts are useful for hypothesis generation, but estimated occupancy and causal direction selection vary materially with [fit corpus, sample size, solver, model checkpoint, or control geometry]. We provide validated estimators, conformance tests, and reporting standards that separate robust effects from sparse-frame artifacts.

This is publishable if the assay repair reveals substantial instability. Do not force a global-workspace narrative if the methods result is stronger.

### 17.4 Outcome D: frozen content-channel result survives

Possible title direction:

> **Prompt-Selected Verbalizable Directions Form a Causal Content Channel Across Open Language Models**

Claim template:

> Prompt-selected J-aligned projectors selectively reduce the probability of the corresponding factual content relative to rank-, energy-, spectrum-, pool-, output-, and geometry-matched controls, while leaving preregistered general-processing metrics intact. The effect [does or does not] distinguish compositional from direct retrieval and [does or does not] depend on the mean Jacobian beyond logit alignment.

This can stand independently of a global-workspace replication, but only after the full control family and phase factorial.

### 17.5 Outcome E: externalization result survives

Possible title direction:

> **Visible Reasoning Changes the Use of Verbalizable Internal State**

Claim template:

> In matched checkpoints or modes, visible reasoning selectively compensates for a preregistered internal content intervention beyond equal-compute and irrelevant-rationale controls. The compensation is associated with re-entry or re-derivation of the target content and improves final-answer behavior, not merely answer mention during reasoning.

This requires the repaired rescue endpoint and matched OLMo/Qwen contrasts.

---

## 18. Concrete edits to the current Part 2 plan

This section maps directly onto `jspace_interp_part2_plan1.md` so an agent can update it without interpreting the entire review.

### 18.1 Replace Section 0

Delete the current opening claim that geometry, static null, frozen specificity, and CoT rescue are established. Replace it with:

> **Current evidential state.** Lab 37 established a functioning open-model J-lens pipeline and generated several provisional descriptive, causal, and temporal findings. A forensic audit found that the broad live ablation omitted the paper's intended-output protection, the capacity comparison used different estimands from the paper, the frozen intervention lacks geometry- and item-energy-matched controls, and the CoT rescue endpoint measures answer occurrence rather than reliable final-answer recovery. Part 2 therefore begins with a mandatory assay-repair and conformance phase. Cross-model conclusions begin only after this phase passes. Existing results remain valuable exploratory priors and engineering baselines, but they are not used as confirmatory evidence.

### 18.2 Insert a new Workstream R before Workstream A

```markdown
## Workstream R: Assay Repair and Publication Conformance

R0. Freeze revisions, manifests, prompt hashes, raw-result schema, and repair preregistration.
R1. Reproduce the paper's output-protected dynamic top-10 J ablation exactly.
R2. Implement paper-defined occupancy and random-adjusted variance.
R3. Replace QR with numerical-rank-safe projectors and log effective rank.
R4. Split prefill and decode interventions factorially.
R5. Add full-sequence scoring, paired clustered inference, and equivalence tests.
R6. Validate sparse pursuit, covariance resume, and report regeneration on synthetic and tiny-model tests.
R7. Re-audit OLMo and Qwen Part 1 headline cells with the repaired assay.

Gate: Workstream A cannot launch until R1 through R6 pass and a pilot positive control succeeds.
```

### 18.3 Replace the Core Battery

The current Core Battery should become:

1. immutable model/tokenizer/lens manifest;
2. recipient-fitted lens plus optional transfer condition;
3. hidden lens validation and negative controls;
4. paper-defined occupancy and excess variance;
5. exact released multihop/selectivity positive controls;
6. output-protected dynamic ablation with matched dynamic controls;
7. static shared-span auxiliary grid with equivalence analysis;
8. frozen J, logit, J-rotated, label-shuffled, and item-matched controls;
9. prefill/decode factorial;
10. hard one-hop and parametric-load tasks;
11. full-sequence log probability, final-answer accuracy, and general-processing guards;
12. raw per-item logging and paired clustered analysis;
13. deterministic primary run plus multi-seed sampled secondary behavior;
14. complete trace capture for any temporal study.

### 18.4 Rewrite A0

Remove “a pass eliminates two five-hour fits.” Replace with:

> Lens transfer is an experimental condition. A small recipient fit is required to evaluate transfer. Exploratory sweeps may use a transferred lens only after equivalence against the recipient lens on readout and projector geometry; confirmatory headline cells use recipient-fitted canonical lenses.

### 18.5 Rewrite A1

Pin exact lineage. Preferred endpoints are OLMo-3.1-32B-Think, OLMo-3.1-32B-Instruct, and their documented base, with intermediate SFT/DPO checkpoints where available. Describe this as a matched-lineage natural experiment, not a one-variable controlled experiment.

### 18.6 Rewrite A2

Split into:

- A2a, Qwen3.6 same-checkpoint official thinking on/off;
- A2b, separate Qwen lineage or base checkpoint, if available and compatible.

Do not call raw completion prompting “non-thinking.”

### 18.7 Expand A3

Add text-path, cache, local/global attention, multimodal bypass, residual hook, derivative, and thinking-mode conformance tests. Require a recipient lens.

### 18.8 Rewrite B1

Use nested fits plus independent corpus draws. Replace fixed cosine/Jaccard thresholds with equivalence margins justified by pilot repeatability and smallest meaningful effect.

### 18.9 Expand B3

Add J-rotated, label-shuffled, local-J, and layer-shuffled controls. Keep logit control as mandatory.

### 18.10 Rewrite B4

Define doses by effective rank and nested bases. Stop escalating dose when general-processing damage crosses a preregistered boundary.

### 18.11 Expand C2

Include exact released multihop, selectivity, capacity, and dual-task tasks at minimum. Document every model-specific adaptation before opening outcomes.

### 18.12 Rename D

Rename “occupancy index” to “output-alignment analysis” to avoid collision with the paper's occupancy term. Make protected versus unprotected ablation its main causal validation.

### 18.13 Replace Section 6 statistics

Use Section 12 of this addendum. In particular, replace:

- arbitrary `n >= 60` with simulation-based power;
- separate bootstraps with paired clustered inference;
- CI overlap with direct contrast and equivalence tests;
- “two seeds” with explicit lens-fit, control-seed, and decoding replication;
- giant matrix BH-FDR with a small Holm-corrected primary family and separate secondary families.

### 18.14 Replace priority order

Use Section 15.1. Preserve the existing “characterize a positive before adding models” rule.

---

## 19. Required Part 2 deliverables

### 19.1 Protocol and preregistration

- `protocol/PAPER_PROTOCOL_CROSSWALK.md`
- `protocol/INTERVENTION_SPEC.md`
- `protocol/OCCUPANCY_SPEC.md`
- `protocol/SCORING_SPEC.md`
- `preregistration/REPAIR_PREREGISTRATION.md`
- `preregistration/SCIENTIFIC_PREREGISTRATION.md`
- `preregistration/DEVIATIONS.md`

The crosswalk should quote no long passages. It should map each paper method to code module, config field, test, and output artifact.

### 19.2 Code and tests

- typed model adapters for OLMo, Qwen, and Gemma;
- rank-safe projector utility;
- output-protected selector;
- phase-controlled hook engine;
- validated sparse solver;
- streaming moments/PCA state;
- full-sequence scorer;
- raw record writer;
- paired analysis package;
- unit and integration tests;
- environment lock and container specification.

### 19.3 Assay-validation report

`reports/ASSAY_VALIDATION.md` should state:

- which Part 1 implementations matched or differed from the paper;
- synthetic and tiny-model test results;
- positive and negative control results;
- old versus repaired OLMo/Qwen capacity values;
- old unprotected versus new protected live ablation;
- frozen control-family results;
- any Part 1 claims upgraded, downgraded, or invalidated.

This report is a publication asset. It demonstrates scientific self-correction.

### 19.4 Raw and derived data

- one Parquet file per experiment shard;
- immutable selection/projector artifacts;
- all traces for preregistered temporal studies;
- aggregate summary JSON generated from raw data;
- figure source tables;
- artifact checksums and provenance graph.

### 19.5 Model-study reports

- `reports/OLMO_LINEAGE.md`
- `reports/QWEN_MODE.md`
- `reports/GEMMA_ARCHITECTURE.md`
- `reports/COT_TEMPORAL_AND_RESCUE.md`
- `reports/BROADCAST.md`, if run
- `reports/FINAL_SYNTHESIS.md`

### 19.6 Claim ledger

Each claim needs:

- exact estimand;
- evidence tier;
- model and task scope;
- effect estimate and interval;
- primary artifact path;
- preregistration link;
- falsifier;
- known boundary;
- whether independently rerun.

Suggested new ledger namespaces:

- `SL2-M*` for methods/assay claims;
- `SL2-O*` for OLMo lineage;
- `SL2-Q*` for Qwen mode;
- `SL2-G*` for Gemma;
- `SL2-C*` for causal/task results;
- `SL2-T*` for temporal/CoT results.

Do not inherit `SL1` wording without re-evaluating it under the repaired assay.

---

## 20. Recommended paper structure

### 20.1 Main text

1. **Introduction:** the generalization question and why open, checkpoint-rich model families matter.
2. **Audit of the replication assay:** protocol fidelity, capacity estimator, intervention controls.
3. **Methods:** validated lens fitting, model adapters, tasks, interventions, statistics, preregistration.
4. **Part 1 reanalysis:** which exploratory findings survive repair.
5. **OLMo lineage:** primary matched-lineage result.
6. **Qwen mode and Gemma architecture:** secondary generalization axes.
7. **Functional task demand:** load, dual-task, and selectivity effects.
8. **Mechanism:** output alignment, layer/phase localization, frozen content specificity.
9. **Chain-of-thought:** complete-sample temporal results and final-answer rescue.
10. **Discussion:** positive replication, bounded null, or method limitation, with explicit scope.
11. **Limitations and falsifiers.**
12. **Reproducibility statement.**

### 20.2 Main figures

A strong paper should be able to tell the story in approximately six main figures:

1. **Protocol crosswalk:** published intervention versus repaired implementation, with validation controls.
2. **Capacity:** paper-defined occupancy and excess variance across OLMo lineage, Qwen, and Gemma.
3. **Primary causal result:** protected dynamic effect versus controls, reasoning tasks versus general text.
4. **Post-training or mode interaction:** OLMo lineage and Qwen thinking on/off.
5. **Task demand:** intervention effect by parametric load at matched baseline difficulty.
6. **Mechanistic or CoT result:** output alignment or confirmatory rescue, depending on which survives.

Static-span, frozen-control breadth, fit-size curves, solver validation, and detailed layer sweeps can live in appendices unless they become the main result.

### 20.3 Tables

- model/checkpoint/lens manifest;
- task families and independent units;
- primary estimands and equivalence margins;
- headline effects with paired intervals;
- preregistration deviations;
- claim ledger summary;
- reproduction status by artifact.

---

## 21. Definition of done

Part 2 is ready for external peer review only when all applicable boxes are checked.

### Protocol fidelity

- [ ] Paper's clean-output protection is implemented and tested.
- [ ] Paper-defined occupancy and excess variance are implemented.
- [ ] Exact paper positive-control tasks run or adaptations are documented before outcomes.
- [ ] Broadcast claims use paper-comparable assays or are explicitly labeled different.

### Numerical validity

- [ ] Sparse solver passes synthetic recovery tests.
- [ ] Interrupted and uninterrupted moments/PCA agree.
- [ ] Projectors use numerical-rank-safe bases.
- [ ] Effective rank and singular spectrum are logged.
- [ ] OLMo and Qwen capacity are recomputed with one final algorithm.

### Causal validity

- [ ] Interventions are phase-resolved.
- [ ] Controls match energy and effective rank on evaluated items.
- [ ] Geometry-preserving and output-alignment controls are included.
- [ ] Positive controls succeed.
- [ ] General-processing damage stays within preregistered limits.
- [ ] Full-answer sequence, not only first token, is scored.

### Statistical validity

- [ ] Raw paired item outcomes are available.
- [ ] Template/family clustering is modeled.
- [ ] Sample sizes come from power simulation.
- [ ] Nulls use equivalence bounds.
- [ ] Primary multiplicity family is frozen.
- [ ] Checkpoint-by-intervention interactions are tested directly.
- [ ] Human grading is blinded and audited where needed.

### Temporal and CoT validity

- [ ] All traces in the confirmatory sample are saved.
- [ ] Lead detector and foil calibration are preregistered.
- [ ] Final post-reasoning answer is the primary rescue endpoint.
- [ ] Thinking, filler, shuffled, correct-rationale, and wrong-rationale arms use matched budgets.
- [ ] The same immutable projector is used across comparison arms.

### Reproducibility

- [ ] Code commit, model revision, tokenizer, data, lens, and environment are pinned.
- [ ] Per-item raw data regenerate every table and figure.
- [ ] No required artifact exists only on an undocumented Drive path.
- [ ] Tests run in CI.
- [ ] A fresh operator reproduces the primary table.
- [ ] Preregistration and deviations are public.
- [ ] Part 1 errata are public.

### Interpretation

- [ ] Claims distinguish readout, sparse geometry, causal necessity, working memory, and global-workspace analogy.
- [ ] Open-model results are not presented as a direct refutation of Claude.
- [ ] “No significant difference” is not called equivalence.
- [ ] “Answer mentioned” is not called task rescue.
- [ ] “Same code path” is not claimed when adapters or estimators differ.
- [ ] Exploratory analyses are visibly labeled.

---

## 22. Final recommendation

Proceed with Part 2, but do not execute the current plan in its present order.

The highest-value next experiment is **not yet OLMo Instruct**. It is the repaired, output-protected paper intervention on the existing OLMo setup, paired with the paper-defined capacity estimator and a valid positive control. That experiment decides what the OLMo Instruct comparison will mean.

Once the assay passes, the OLMo base/Think/Instruct lineage is the strongest core study. It can test whether post-training changes the role of verbalizable internal state using public checkpoints with a common lineage. Qwen's official thinking toggle then supplies a distinct within-weight inference-mode contrast. Gemma 4 provides an architecture and multimodal/hybrid-attention contrast after the primary result is banked.

The likely publication is no longer merely “we tried the paper on OLMo.” It can become one of four sharper contributions:

- a faithful open-model replication with post-training localization;
- a preregistered boundary-of-generalization null;
- a methods paper that exposes where J-lens conclusions depend on estimator and intervention choices;
- a rigorously controlled causal content-channel result that is distinct from the global-workspace claim.

Any of those outcomes would be worthwhile. The common requirement is that Part 2 first turn the current assay from a clever exploratory instrument into a calibrated scientific measuring device. Right now the telescope has found several strange lights, but some of the constellations are still lens flare. The repair phase is what makes the next sky map publishable.

---

## Appendix A. Source references

### Target lab

- [Target merge commit](https://github.com/karlb-dev/labs/commit/4097c44713d0084d4da3d3c084e79aed2068c740)
- [Lab 37 report](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/report/REPORT_v2.md)
- [Lab 37 specification and claim ledger](https://github.com/karlb-dev/labs/blob/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/labs/lab37_jspace_workspace.md)
- [Lab 37 code](https://github.com/karlb-dev/labs/tree/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/code)
- [Lab 37 result summaries](https://github.com/karlb-dev/labs/tree/4097c44713d0084d4da3d3c084e79aed2068c740/interpretability/jspaces/phases/phase1/results)

### Paper and reference implementation

- [Gurnee et al., “Verbalizable Representations Form a Global Workspace in Language Models”](https://transformer-circuits.pub/2026/workspace/index.html)
- [Anthropic Jacobian-lens reference implementation](https://github.com/anthropics/jacobian-lens)
- [Reference fitting estimator](https://github.com/anthropics/jacobian-lens/blob/581d3986/jlens/fitting.py)
- [Released experiment-set descriptions](https://github.com/anthropics/jacobian-lens/blob/581d3986/data/experiments/README.md)
- [Neel Nanda, “A Review of Anthropic's Global Workspace Paper”](https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S/a-review-of-anthropic-s-global-workspace-paper)

### Proposed model matrix

- [OLMo-3.1-32B-Think](https://huggingface.co/allenai/Olmo-3.1-32B-Think)
- [OLMo-3.1-32B-Instruct](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct)
- [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B)

## Appendix B. One-page agent checklist

1. Do not launch the model matrix first.
2. Inventory and hash all old artifacts.
3. Pin model, tokenizer, data, code, and environment revisions.
4. Implement clean-output-protected dynamic ablation.
5. Implement paper-defined occupancy and random-adjusted variance.
6. Replace QR with rank-safe SVD bases.
7. Split prefill and decode intervention phases.
8. Score full answer sequences.
9. Store raw paired per-item results.
10. Add J-rotated, label-shuffled, logit, and item-matched controls.
11. Validate on synthetic data and a tiny model.
12. Rerun the decisive OLMo pilot and positive controls.
13. Write scientific preregistration and freeze the holdout.
14. Run OLMo base/Think/Instruct before Qwen or Gemma breadth.
15. Run Qwen official thinking on/off as a within-checkpoint contrast.
16. Run Gemma only after adapter conformance.
17. Use power simulation, clustered inference, and equivalence tests.
18. Save every confirmatory CoT trace and grade final answers.
19. Reproduce the primary table on a fresh machine/operator.
20. Publish Part 1 errata and all deviations alongside the final report.
