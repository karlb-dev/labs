# jspace_lab_nextsteps_4_3.md

## J-space Phase 4, development block 3: finish Qwen lens convergence, close the freeze design, and integrate the parallel mechanism tracks

**Review target:** `karlb-dev/labs`, branch `interp_jspace_part2`, reviewed at the clean branch boundary `3b041735d8b842de46a9c0a474fccd0c44e0841a`, together with the canonical 2026-08-01 Phase 4.2 handoff, the Phase 4 candidate preregistration v0.9, the registered OLMo lineage analyses, the Qwen A120/A250/A500 fit and functional-gate series, Bank B and Bank W methods artifacts, and the current paper conclusion skeleton.

**Phase decision:** remain in **Phase 4** and name this plan `jspace_lab_nextsteps_4_3.md`. Phase 4 is not waiting for more vague exploration. It has three precise pre-freeze obligations left: finish the Qwen canonical-lens decision, make the P4-P1 and P4-P2 designs statistically executable, and close the P4-P3 capability population. The parallel OLMo-lineage and Gemma work should live in separate namespaces and branches. Their results may enter Phase 4 only through hash-pinned import manifests after they are complete enough to review.

**Current dynamic boundary:** the active Phase 4 VM owns the Qwen draw-A continuation from n=500 toward n=1000. The last durable boundary in the supplied handoff is n=530, but this plan must always trust the highest valid `checkpoint_state.json` found at launch rather than this static number. A500 is registered and immutable. A250-to-A500 structural convergence passes, but functional selection stability, selected-ID stability, and bridge-rescue stability fail the precommitted thresholds. The frozen router therefore selected Branch B and continued to n=1000. No Phase 4 confirmatory or replication intervention outcome has been opened.

> **Paste-line for the coding/research agent**
>
> Read `jspace_lab_nextsteps_4_1.md`, its accepted addendum, `jspace_lab_nextsteps_4_2.md`, its accepted addendum, `PHASE3_STATE_OF_RECORD.md`, `PHASE4_DEVELOPMENT_REPORT.md`, `SCIENTIFIC_PREREGISTRATION_PHASE4_CANDIDATE.md`, and the newest canonical `INPROGRESS_VM12_20260801.md` before changing code. Phase 3 is immutable. Phase 4 remains development-only. The current Qwen A1000 continuation has exclusive ownership of its recovery lock: do not launch a duplicate process, do not delete or relocate the Qwen cache, do not alter the nested corpus or fit contract, and do not register a partial lens milestone. Let the existing process bank atomic checkpoints until reclaim. On the next VM, recover the highest valid checkpoint and finish A1000 under the unchanged producer. Then execute the frozen A500-to-A1000 structural and functional decision queue, add the selection-margin audit in this plan, and make one explicit canonical-lens decision. After that, complete the Qwen mode variance pilot and power design, import the two OLMo Bank-W capability gates from the separate lineage track, settle Bank B by the pre-outcome branch rule below, verify the whole registry, and stop at a reviewable Phase 4 freeze candidate. Never self-sign independent-review or PI fields, never expose untouched outcomes before the tagged freeze, and supersede rather than overwrite every correction.

---

# 0. Executive verdict

## 0.1 The new result is a structural-function split, not merely “the small lens is noisy”

The A250-to-A500 series sharpened the Qwen measurement problem.

At the matrix and token-direction level, the two nested same-corpus lenses are extremely close in the frozen L20-L44 assay band. Raw, identity-subtracted, and scaled-identity-subtracted matrix cosines are all near one. Task-token row cosines also pass by a wide margin. Nevertheless, three functionally important quantities do not pass:

- normalized selected-projector overlap remains about 0.70 rather than the frozen 0.85 floor;
- selected-token Jaccard remains about 0.54 rather than the frozen 0.75 floor;
- the bridge-rescue estimate moves by more than the frozen 0.25-nat ceiling.

Meanwhile, occupancy, centered excess, the span-safe-specific point estimate, the tail rate, the G4 positive control, bridge preference, and the structural dependency all pass their point-estimate rules.

That is not a single undifferentiated “instability.” It says:

1. the average transport operators have largely converged;
2. sparse top-k selection remains discontinuous or underidentified;
3. most aggregate causal conclusions are much more stable than the exact selected rows;
4. at least one mechanism endpoint, bridge rescue, remains sensitive enough to block a canonical-lens declaration.

A1000 is therefore justified. It must also be the **last automatic fit-size escalation** in Phase 4. If exact sparse selections still fail at n=1000, the correct response is not an endless n=2000 fit. The project must decide whether the scientific estimand requires literal token-row identity, a stable selected subspace, or a stable downstream causal quantity.

## 0.2 A1000 can close the fit-size question in more than one way

There are four legitimate outcomes after A1000:

### Outcome Q-L1: full convergence

A500-to-A1000 and A1000-to-published both pass structural, selected-row, selected-subspace, capacity, positive-control, and causal gates. A1000 becomes the canonical Qwen Phase 4 lens. The mode pilot and any later Phase 4 Qwen development use its registered hash.

### Outcome Q-L2: row identities drift, subspaces and causal quantities converge

Literal selected-ID Jaccard remains below the original floor, but projector overlap, principal angles, protected-span geometry, capacity, G4, span-safe specificity, tail membership, bridge preference, and bridge rescue become equivalent within frozen tolerances. This means the overcomplete token dictionary has near-tied aliases or interchangeable rows. The project should not keep fitting until arbitrary row labels agree. It should replace literal selected-ID stability as a **methods diagnostic**, while defining the scientific object as the selected span and its causal consequences. This change must be prospective, reviewed, and documented before the Phase 4 freeze.

### Outcome Q-L3: sparse subspaces drift, causal endpoints remain stable

Selected IDs and selected projectors fail, but the causal endpoints pass robustly. That supports a “many sparse coordinate realizations, stable causal channel” result. A single canonical sparse dictionary is not licensed. The primary Phase 4 intervention must then use a cross-fit or ensemble definition that is frozen before untouched outcomes, such as a consensus span or a lens-random-effect analysis. The published-lens Phase 3 results remain intact because those were already frozen and replicated under their own instrument.

### Outcome Q-L4: causal conclusions move materially

One or more key causal endpoints remain decision-sensitive at A1000. A single fixed Qwen lens is not a stable Phase 4 instrument. P4-P2 and any Qwen primary that relies on a newly fitted lens remain blocked. The methods result becomes the headline: the averaged Jacobian operator can appear structurally converged while sparse causal interpretation remains fit-sensitive. Phase 4 must either bind to the external published n=1000 lens for comparability, use a preregistered cross-fit estimator, or remove the affected Qwen primary.

The branch table in Section 5 makes these outcomes executable.

## 0.3 The remaining freeze blockers are now asymmetric

The three proposed primaries are not equally mature.

- **P4-P3, controlled load, is closest to executable.** Bank W is authored, audited, split, and powered. Qwen passes capability. The two OLMo capability gates and the joint common-support calculation remain. These can run independently on the OLMo-lineage VM and be imported without exposing an intervention outcome.
- **P4-P2, Qwen mode by phase, has passed its parser and baseline gate.** Its remaining design uncertainty is variance and power. The consumed-family variance pilot is lawful only after a canonical-lens decision and producer review.
- **P4-P1, bridge specificity, is not currently a viable confirmatory primary.** The verified 40-family bank is far too small under the observed heavy-tailed variability. Reallocating the same families does not fix it. Phase 4 must choose between estimation-only, a substantively lower-variance orthogonalized estimand, or a much larger bank.

The project should not delay the freeze indefinitely to preserve a three-primary symmetry that the data do not support. The default recommendation in this plan is:

> Keep P4-P3 as a primary, retain P4-P2 only if the consumed-family pilot yields a feasible powered design, and reclassify P4-P1 as a preregistered estimation-and-replication target unless an answer-direction-orthogonal development estimand passes a strict variance gate.

That is a stronger paper than an underpowered three-test family.

## 0.4 Parallel work is useful only with namespace discipline

The user plans three simultaneous VMs:

1. the main Phase 4 continuation;
2. a dedicated OLMo-lineage program;
3. a dedicated Gemma transport autopsy.

This is sensible. It becomes dangerous if all three edit the same registry, preregistration, reports, or Drive root. Section 2 freezes a branch and import contract so parallelism buys information rather than merge archaeology.

---

# 1. State of record at the Phase 4.3 boundary

## 1.1 Immutable prior evidence

Treat these as read-only inputs:

- Phase 2 completion and its confirmatory/replication artifacts;
- Phase 3 completion tag `jspace-phase3-complete-v1`;
- Phase 3 state-of-record and release manifest;
- every registered Phase 4 event through the clean boundary at `3b041735...`;
- A120, A250, and A500 Qwen lenses and their registered manifests;
- the frozen nested draw-A and draw-B corpora;
- the OLMo base/3.0 Think/3.1 Think/3.1 Instruct development grids;
- the common-support lineage analysis;
- Bank B candidate v2, source verification, power, and design-feasibility outputs;
- Bank W candidate v2, capability protocol, power ruler, and partition;
- Qwen official-mode parser v2 and passing model-backed baseline.

Do not repair prior artifacts in place. Any new interpretation, threshold change, or metadata correction receives a new event and an explicit supersession edge.

## 1.2 Current Qwen fit lineage

The nested draw-A sequence is cumulative and must remain so:

```text
A120 ⊂ A250 ⊂ A500 ⊂ A1000
```

The current producer contract is immutable through A1000:

- model: exact Qwen3.6-27B revision already pinned in Phase 4;
- corpus: registered draw-A JSONL and exact order;
- target layer: 63;
- source layers: 0 through 62;
- maximum sequence length: 128;
- skip-first: 16;
- estimator: upstream `jlens` revision already pinned;
- runtime packages: exact PyTorch, Transformers, Triton, FLA versions in the config;
- 48 fused linear-attention blocks required;
- cumulative float32 checkpoint;
- fp16 registered milestone lens;
- three-prompt atomic checkpoint boundary;
- no CPU model fallback.

A1000 is not a fresh refit. It is the continuation of the exact cumulative estimator that produced A120, A250, and A500.

## 1.3 Current A1000 process contract

While the current VM is alive:

1. Do not start any second process that can write the draw-A recovery directory.
2. Do not run `git pull`, switch branches, or edit tracked files underneath the clean producer process unless the runner is explicitly designed for it. Prefer documentation in a separate worktree if absolutely necessary.
3. Do not remove the local Qwen model, FLA environment, checkpoint inode, or Drive recovery pair.
4. Monitor the log, lock, and `checkpoint_state.json` listed in the handoff.
5. Treat the highest complete header/checkpoint pair as truth. An in-flight chunk is expendable.
6. Do not create an evidence event for n=533, n=835, or any other partial count. Partial state is recovery state, not a scientific lens milestone.
7. At impending reclaim, wait for the current three-prompt chunk to synchronize if feasible, then update only the dynamic handoff with the highest valid boundary and hashes.

## 1.4 What the existing functional gate actually established

The A250-to-A500 functional gate contains several scientifically different classes of metric:

### Operator and task-row structure

These pass strongly. They show that the mean transport map and sampled task-token directions are converging.

### Sparse selection identity

Selected-ID Jaccard fails. This is sensitive to near ties, dictionary redundancy, top-k boundary movement, and token aliases.

### Sparse selected subspace

Normalized projector overlap fails, although median principal angles may remain much smaller than the worst-case angle. This says the selected span itself is not yet stable enough under the frozen rule.

### Distributional causal effects

Mean span-safe specificity and tail rate pass their point-estimate thresholds, but the mean-difference confidence interval does not establish formal equivalence. This is suggestive, not closure.

### Mechanism endpoints

Bridge preference passes, while bridge rescue fails. This difference is important. A preference endpoint can be stable while a protection/rescue endpoint changes because the selected span intersects the true bridge representation differently.

The A1000 queue must preserve these classes rather than collapse them into a single PASS/FAIL cell.

---

# 2. Parallel-branch and artifact contract

## 2.1 Branches

Use three branches cut from the exact clean Phase 4 boundary:

```text
main Phase 4:  interp_jspace_part2
OLMo lineage: interp_jspace_olmo_lineage
Gemma autopsy: interp_jspace_gemma_transport
```

The side branches should be created from `3b041735d8b842de46a9c0a474fccd0c44e0841a` unless a later clean main commit contains only documentation or infrastructure that the side track explicitly needs. Record the exact parent SHA in each side-track foundation manifest.

## 2.2 Repository namespaces

```text
interpretability/jspaces/phases/phase4/           # main Phase 4 only
interpretability/jspaces/sidelines/olmo/      # OLMo side track only
interpretability/jspaces/sidelines/gemma/             # Gemma side track only
```

The OLMo and Gemma agents may import code from `jspace_phase4`, `jspace_phase3`, or `jspace_part2`, but must not edit those packages while working in parallel. If a reusable bug fix is necessary:

1. implement it in a small isolated compatibility module in the side namespace;
2. add a conformance test against the original implementation;
3. later upstream it through a dedicated reviewed commit after the active Phase 4 process is safe.

## 2.3 Drive roots

```text
Phase 4: /content/drive/MyDrive/interpret/special-lab-1/phase4_20260731
OLMo:    /content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_<date>
Gemma:   /content/drive/MyDrive/interpret/special-lab-1/gemma_transport_<date>
```

No side-track process may write into the Phase 4 run root. Read-only materialization by exact hash is allowed.

## 2.4 Registries

Each namespace gets its own append-only registry:

```text
jspace_phase4/reports/evidence_events.jsonl
jspace_olmo_lineage/reports/evidence_events.jsonl
jspace_gemma/reports/evidence_events.jsonl
```

Evidence IDs receive prefixes:

```text
p4-...
ol-...
gm-...
```

The Phase 4 registry must reject native `ol-` or `gm-` events. Later imports use one explicit event per side-track release bundle.

## 2.5 Import bundle schema

Each side track eventually emits:

```text
release/IMPORT_BUNDLE_PHASE4.json
release/IMPORT_BUNDLE_PHASE4.md
```

Minimum JSON fields:

```json
{
  "schema_version": 1,
  "source_study": "jspace-olmo-lineage",
  "source_branch": "interp_jspace_olmo_lineage",
  "source_commit": "<sha>",
  "source_registry_sha256": "<sha256>",
  "tier": "development",
  "claim_boundary": "<plain language>",
  "inputs": [
    {"logical_uri": "...", "sha256": "...", "bytes": 0}
  ],
  "outputs": [
    {"logical_uri": "...", "sha256": "...", "bytes": 0,
     "phase4_use": "capability gate or methods sensitivity"}
  ],
  "tests": {"passed": 0, "failed": 0},
  "forbidden_uses": ["confirmatory evidence", "PI sign-off"]
}
```

The main Phase 4 importer must:

- verify the source commit is reachable;
- verify the side registry hash;
- rehash every imported output;
- verify no side output overlaps untouched Phase 4 families;
- preserve the source tier;
- refuse an import whose source worktree was dirty at production time;
- write a Phase 4 import event rather than copying native side events into the main registry.

## 2.6 Merge discipline

- Side tracks do not edit `PHASE4_DEVELOPMENT_REPORT.md`, Phase 4 preregistration, Phase 4 figures, or Phase 4 registry.
- Main Phase 4 does not edit side-track reports.
- Merge side branches with ancestry preserved after their release bundle is ready.
- Resolve import and narrative integration in a small main-branch commit.
- Never squash away the source branch’s scientific boundary unless the release manifest separately pins every source commit.

---

# 3. Immediate instructions for the current Phase 4 VM

## 3.1 Let A1000 own the machine

The current VM has one job: continue A1000 safely. Do not attempt the OLMo Bank-W gates, the Qwen mode pilot, or a new Bank B experiment on this machine while the Qwen process is active.

A model-fit continuation at roughly 63 GB allocation plus recovery copies is not a good neighbor for unrelated model work. The expected value of one extra side experiment is lower than the risk of corrupting or evicting a 500-prompt cumulative fit.

## 3.2 Monitoring loop

Use a lightweight monitor that never touches model state:

```bash
set -euo pipefail

STATE=/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/lens/qwen36-27b/nested_fit/draw_a/recovery/checkpoint_state.json
LOG=/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/qwen_frozen_branch_followup_20260801.log
LOCK=/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/qwen_frozen_branch_followup.lock

while true; do
  date -u
  test -f "$STATE" && python - <<'PY'
import json
from pathlib import Path
p = Path("/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/lens/qwen36-27b/nested_fit/draw_a/recovery/checkpoint_state.json")
print(json.dumps(json.loads(p.read_text()), indent=2))
PY
  test -f "$LOCK" && cat "$LOCK" || true
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
  tail -n 20 "$LOG" || true
  sleep 900
done
```

This monitor is illustrative. Reuse the repository’s existing watchdog if it already records equivalent information.

## 3.3 Reclaim handoff

At the final safe boundary:

1. Verify the latest Drive header and checkpoint are a complete pair.
2. Verify checkpoint size and SHA against the header.
3. Record `next_idx`, checkpoint hash, checkpoint bytes, fit-contract hash, active code commit, log path, and lock state.
4. Confirm no registered A1000 event exists yet.
5. Confirm the branch remains clean or explain any documentation-only commit.
6. Push the handoff.
7. Do not intentionally kill a healthy process merely to produce a prettier boundary unless reclaim is actually imminent and the current chunk has synchronized.

---

# 4. Next-VM A1000 recovery and completion

## 4.1 Bootstrap

From a fresh host GPU process:

```bash
cd /content/labs
git fetch origin
git checkout interp_jspace_part2
git pull --ff-only

git status --short  # must be empty
cd interpretability/jspaces/phases/phase4
python -m pytest -q
```

Then restore the exact runtime and model snapshot specified by `configs/p4_qwen_nested_lens_fit_dev.yaml`. Do not upgrade packages because a newer wheel exists.

## 4.2 Recovery gate

Before model load, the producer must:

- run the same-process CUDA FP16 matmul gate;
- verify the exact model snapshot manifest and all weight hashes;
- verify the exact `jlens` revision and clean source tree;
- verify all runtime package versions;
- verify FLA availability and all 48 fused block bindings;
- compare local and Drive recovery headers;
- choose the highest compatible valid boundary;
- rehash the selected 6.6-GB checkpoint;
- refuse mismatched fit contracts rather than beginning a second estimator.

Launch only the existing producer:

```bash
python -m jspace_phase4.experiments.p4_qwen_nested_lens_fit \
  --config configs/p4_qwen_nested_lens_fit_dev.yaml \
  --draw draw_a \
  --stop-at 1000
```

or the exact repository wrapper if it additionally validates the frozen branch decision.

The startup log must show `recovered_next_idx` equal to the highest durable boundary, not automatically 500 and not the last remembered number in this plan.

## 4.3 A1000 registration gate

Register `p4-qwen-lens-fit-drawA-n1000-dev-v1` only when all of the following hold:

- `lens.n_prompts == 1000`;
- exact nested draw-A corpus hash matches;
- all source layers 0 through 62 are present;
- all tensors are finite and have expected shape/dtype;
- final checkpoint and fp16 lens hashes are independently recomputed;
- local and Drive copies match;
- all 1000 prompt diagnostic rows are present and finite or explicitly classified by a precommitted skip rule;
- no prompt has been trimmed post hoc;
- input manifest includes the cumulative fit contract and recovery boundary;
- registry creation occurs from a clean tree;
- the event and outputs pass `repro4` verification after registration.

## 4.4 Influence audit for prompt 323

Prompt 112 was audited because it dominated the A120 prompt-norm distribution. A500 introduces prompt 323 with an even larger recorded maximum. Do not silently assume that A500 dilution makes it harmless.

Create:

```text
jspace_phase4/experiments/p4_qwen_lens_influence_prompt323.py
configs/p4_qwen_lens_influence_prompt323_dev.yaml
```

The audit should:

1. verify the equal-weight running-mean contract on a tiny model and from adjacent cumulative checkpoints;
2. reconstruct the exact A500 leave-one-out estimator algebraically;
3. when A1000 is available, compare A1000 to an influence approximation that removes prompt 323’s contribution while retaining all other prompts;
4. report raw, minus-identity, and minus-scaled-identity operator changes;
5. report answer, bridge, shared-token, and uniform-row changes;
6. evaluate selected-ID, projector, capacity, G4, span-safe, and bridge endpoints on the frozen functional subset if the structural effect exceeds a precommitted materiality threshold;
7. retain the prompt regardless of the result.

This is a sensitivity analysis, not a trimming license.

---

# 5. A1000 structural and functional decision queue

## 5.1 Freeze successor configs before opening A1000 comparisons

Before the A1000 lens is registered, commit configs that name logical placeholders rather than an unknown future hash. Immediately after registration, create a hash-binding commit that changes only the A1000 hash and registered URI, exactly as the A500 queue did.

Required successors:

```text
p4_qwen_lens_convergence_drawA_n500_n1000_dev.yaml
p4_qwen_multilens_functional_gate_a500_a1000_dev.yaml
p4_qwen_selection_margin_a500_a1000_dev.yaml
p4_qwen_lens_influence_prompt323_dev.yaml
```

## 5.2 Structural comparison

Use the exact same fixed token-ID sample and Rademacher probes used by the A120-to-A250 and A250-to-A500 studies. The evidence ID must not silently generate a new sample.

Report at every layer and for the L20-L44 assay band:

- raw matrix cosine and symmetric relative Frobenius delta;
- `J-I` cosine and delta;
- `J-alpha I` cosine and delta;
- Jacobian-row cosine quantiles;
- task-token direction cosine quantiles for answer-only, bridge-only, and shared IDs;
- centered linear CKA;
- random transport-probe cosine and relative error;
- incremental block estimator versus cumulative estimator;
- identity scale and identity fraction;
- full-attention versus linear-attention layer strata if relevant to Qwen’s architecture.

The existing structural floors remain diagnostics. Do not change them after seeing A1000.

## 5.3 Functional gate

Run the exact same item sets, aliases, condition order, baseline caches, seeds, and endpoints as the A250-to-A500 gate. Compare at minimum:

```text
published n=1000
A500
A1000
```

A120 and A250 may be included in the figure for trajectory context, but the frozen branch decision should use A500-to-A1000 and A1000-to-published according to the config written before outcomes.

Required metrics:

### Selection geometry

- selected-ID Jaccard;
- rank-biased overlap for top-1 through top-20;
- selected-span normalized projector overlap;
- principal-angle distribution, not only median and maximum;
- protected-span overlap and lost rank;
- per-layer and per-position overlap;
- bridge-token survival;
- answer-direction survival.

### Capacity

- marginal-gain occupancy at the frozen crossing rule;
- centered excess at occupancy;
- random-dictionary controls;
- layer 24/32/40 and full assay-band summaries.

### Positive controls

- G4 selected-J swap flip rate;
- random-swap rate;
- clean baseline rate;
- delivered injection fidelity.

### Causal task endpoints

- span-safe specific mean;
- tail rate at the frozen threshold and threshold curve;
- item/family effect correlation;
- exact paired differences with family intervals;
- formal TOST only where a frozen equivalence margin and power are valid.

### Bridge endpoints

- true-versus-distractor protection rescue;
- original-versus-counterfactual preference;
- generation trichotomy if available;
- answer-direction comparator;
- geometry and selected-span diagnostics.

### Prose

- exact matched-control prose cost;
- span-safe prose cost;
- whether lens changes preserve the task/prose ordering.

## 5.4 Selection-margin audit

The current selected-ID and projector failures may arise from discontinuous top-k selection over nearly tied scores. Add a dedicated audit before deciding that the transport operator itself remains scientifically unstable.

For each frozen item, layer, and position, record under A500 and A1000:

- top 32 token IDs and scores before protection;
- top 32 eligible IDs after protection;
- score gaps `s_k-s_{k+1}` for k in {1, 2, 5, 10, 20};
- achieved rank after numerical-rank filtering;
- pairwise cosine among swapped-in and swapped-out rows;
- whether row swaps are aliases, morphological variants, or semantically unrelated;
- projector change after replacing each disputed row by the opposing lens row;
- causal dose contributed by the disputed rows;
- core-set Jaccard for rows separated from the boundary by a frozen margin.

Define a **selection margin** without looking at behavioral outcomes. One defensible form is:

```python
margin_k = (score[k - 1] - score[k]) / max(abs(score[k - 1]), eps)
```

Then partition positions prospectively into:

- stable-core positions;
- near-tie positions;
- rank-deficient positions.

Report selection and causal stability by stratum. Do not exclude near-tie positions from the primary functional gate. This is a mechanism audit.

## 5.5 Consensus and cross-fit diagnostics

If A500 and A1000 remain row-unstable, compute these development-only alternatives:

### Intersection core

Rows selected by both lenses at a position.

### Union span

Orthonormal span of rows selected by either lens, dose matched to a frozen target rank and energy.

### Score-averaged consensus

Average normalized selection scores across A500 and A1000, then select top-k once.

### Cross-fit lens

Fit selection on one corpus half and estimate effects on families not used for any design choice. Because the lens corpus is generic WikiText rather than task families, this is primarily a lens-random-effect diagnostic, not a behavioral data split.

### Lens-random-effect analysis

Treat lens fit as a second sampling axis. Report effect variance across A120/A250/A500/A1000/published rather than pretending one fit is exact.

These diagnostics do not automatically become Phase 4 primaries. They tell the reviewer whether sparse-coordinate instability is a nuisance, a genuine scientific variable, or a threat to the intervention.

## 5.6 Frozen A1000 branch table

Apply this table without inventing a fifth branch after outcomes.

| Branch | Structural operator | Selected span | Aggregate causal | Bridge endpoint | Action |
|---|---|---|---|---|---|
| Q-L1 | pass | pass | pass/equivalent | pass/equivalent | nominate A1000 canonical |
| Q-L2 | pass | row IDs fail, projector passes | pass/equivalent | pass/equivalent | canonicalize A1000; demote literal ID stability to diagnostic; state dictionary redundancy |
| Q-L3 | pass | projector fails | stable | stable | no single sparse dictionary claim; freeze consensus/cross-fit instrument only after review |
| Q-L4 | pass or fail | any | decision-sensitive | decision-sensitive | Qwen new-lens primaries blocked; choose published lens or remove/redefine primary |
| Q-L5 | fail | fail | any | any | averaged transport not converged; Phase 4 methods result, no new Qwen causal primary |

A branch may not be chosen merely because it preserves the desired P4-P2 schedule.

---

# 6. P4-P2: Qwen official thinking-mode variance pilot

## 6.1 Preconditions

Do not run the intervention pilot until all are true:

- Qwen canonical-lens branch has been resolved;
- the selected lens or consensus instrument has a registered hash and manifest;
- the v2 official-mode baseline event is live and passes;
- the parser and phase-hook goldens pass in the current environment;
- the GPU producer has an independent code review focused on phase ownership, cache behavior, and per-token protection alignment;
- the consumed 20-family subset and eight-cell design exactly match the registered pilot protocol;
- no untouched Phase 4 mode family has been opened.

## 6.2 Producer design

Proposed module:

```text
jspace_phase4/experiments/p4_qwen_mode_variance_pilot.py
```

Each family contributes one composed fact. The frozen cells are:

```text
thinking_on  × prefill      × span_safe_J
thinking_on  × prefill      × matched_control
thinking_on  × final_answer × span_safe_J
thinking_on  × final_answer × matched_control
thinking_off × prefill      × span_safe_J
thinking_off × prefill      × matched_control
thinking_off × final_answer × span_safe_J
thinking_off × final_answer × matched_control
```

For every row, record:

- exact prompt/template tokens and hashes;
- generated tokens and parser states;
- phase hook fire counts by token;
- wrong-phase fire count;
- selected and protected ranks;
- delivered rank and energy;
- protected-span overlap;
- final normalized answer class;
- accepted-alias match;
- full original-answer sequence log probability on generated context;
- parse/truncation/error status;
- generation length and reasoning length;
- baseline cache identity and deterministic replay sentinel.

## 6.3 Analysis discipline

The pilot exists to estimate variance, not to produce a result-shaped SESOI.

Primary planning object per family:

```text
I_f = [damage(on, final) - damage(on, prefill)]
      - [damage(off, final) - damage(off, prefill)]
```

where `damage = quality(control) - quality(J)`.

Report:

- sample SD of family `I_f`;
- bootstrap 90% upper confidence bound for SD;
- atom masses and support for the binary endpoint;
- correlation among the eight cells;
- parse-failure contribution under the frozen “failure = incorrect” rule;
- continuous log-probability interaction as a named design sensitivity;
- no promoted claim about the pilot mean.

The planning SD is the larger of sample SD and its upper bound, as already registered.

## 6.4 SESOI review

After the variance pilot, prepare a blind design memo containing:

- the variance distribution;
- candidate substantive SESOIs stated in final-answer accuracy points and in expected families changed;
- exact or Monte Carlo family sign-flip power under realistic atom distributions;
- Holm planning alpha under the final primary family;
- family counts for confirmatory and replication;
- required baseline common support;
- compute estimate.

The memo should not show the signed pilot mean until the PI has selected the SESOI or declared the endpoint infeasible. The code can produce an encrypted or separately permissioned mean artifact if necessary, but ordinary file separation and explicit review procedure is likely sufficient.

## 6.5 Feasibility branches

| Result | Action |
|---|---|
| Accuracy interaction powered at a substantive SESOI with available bank | retain P4-P2 primary |
| Accuracy too discrete, continuous LP interaction powered and scientifically acceptable | prospectively amend P4-P2 before freeze; accuracy becomes secondary |
| Neither endpoint feasible without hundreds/thousands of families | remove P4-P2 from primary family; keep mode mechanism as estimation-first development/future work |
| Parser or phase hooks fail under intervention | repair methods and rerun consumed pilot only; untouched families remain sealed |
| Lens branch Q-L4/Q-L5 | P4-P2 blocked unless a new reviewed instrument is frozen |

Do not enlarge the SESOI simply to save the primary.

---

# 7. P4-P1: resolve the underpowered bridge primary

## 7.1 The current design cannot be rescued by shuffling the split

The existing 40-family Bank B is scientifically useful and carefully verified. It is not a powered test of a 0.25-nat two-component intersection-union effect under the observed heavy-tailed variability. No 20/10/10 versus 10/15/15 rearrangement fixes that.

The Phase 4.3 plan requires a decision before freeze, not another generic power plot.

## 7.2 Default branch: estimation-only P4-E1

Unless Section 7.3 succeeds, reclassify the bridge-specific study as:

```text
P4-E1: preregistered estimation plus held-out replication
```

Use all available untouched families through a precommitted confirmatory/replication split, report both component estimates and intervals, and prohibit reject/non-reject language. This preserves the scientifically important comparison without placing an impossible p-value inside Holm.

The primary family then contains P4-P2 if feasible and P4-P3. Holm is recomputed over the final count before freeze.

## 7.3 Optional development rescue: answer-direction-orthogonal bridge estimand

The existing bridge vector may share substantial geometry with the counterfactual answer direction. Direct answer-direction injection can therefore reproduce much of the semantic movement. A lower-variance and more specific estimand might operate only on the bridge component orthogonal to the answer span.

For a bridge direction `b` and protected answer basis `Q_a`:

```python
b_perp = b - Q_a @ (Q_a.T @ b)
```

Then normalize and calibrate dose using activation-scale geometry, never answer outcomes.

Development arms on consumed families:

1. counterfactual bridge orthogonal component;
2. geometry-matched unrelated orthogonal component;
3. direct counterfactual-answer direction;
4. full counterfactual bridge;
5. random orthogonal direction;
6. no injection.

Required gates:

- retained bridge norm above a frozen floor;
- answer-span overlap below numerical tolerance;
- rank and delivered-energy match;
- semantic bridge identity remains decodable under a positive readout control;
- no outcome-based sign selection;
- family SD and heavy-tail profile materially improve;
- semantic effect does not collapse entirely.

The estimand is admitted only if an outcome-blind geometry gate and a consumed-data variance gate were written before the development run. Even then, a new untouched bank or larger family count may be required.

## 7.4 Decision rule

```text
if orthogonalized estimand has a substantively meaningful effect target,
   controlled type-I error, and >=0.80 power at the available or newly
   authored family count:
       prospectively replace P4-P1 before freeze
else:
       P4-P1 becomes estimation-only P4-E1
```

No third path allows the original underpowered binary endpoint to remain because it is emotionally appealing.

## 7.5 Bank enlargement, only if explicitly chosen

A genuinely larger Bank B must add **new canonical relation families**, not more facts inside the same 40 families. The unit of inference is family. Expansion should target at least the family count implied by the conservative powered design, preserve entity-disjoint partitions, and rerun source verification. If that count is too large for the current paper, say so and use estimation-only.

---

# 8. P4-P3: close the capability population through the OLMo side track

## 8.1 Side-track responsibility

The dedicated OLMo-lineage VM should run only the two baseline Bank-W capability gates and the all-model joint-support calculation needed by Phase 4. It may also run broader OLMo development science in its own namespace, but Phase 4 imports only the capability outputs at this stage.

Required evidence:

```text
ol-bank-w-capability-olmo31-think-dev-v1
ol-bank-w-capability-olmo31-instruct-dev-v1
ol-bank-w-joint-support-dev-v1
```

## 8.2 Import acceptance

The Phase 4 import event verifies:

- exact Bank W candidate v2 hash;
- exact development-family partition hash;
- exact tokenizer/model revisions;
- full 384-row grid per model;
- candidate-set scoring implementation matches Phase 4 protocol;
- no intervention columns exist;
- load 2 and load 6 accuracy gates;
- paired high-minus-low family interval;
- per-model independently capable families;
- joint support across every independently eligible model;
- source registry and output hashes.

## 8.3 Model-set branch

The model set is every independently passing model. Do not drop a passing model because it reduces joint support.

Possible outcomes:

- **All three pass, joint support >=20:** primary max-T includes all three.
- **Qwen plus one OLMo pass, joint support >=20:** primary includes those two; failed model descriptive.
- **Only Qwen passes with >=20:** primary is one-model rather than max-over-models; update power and primary wording before freeze.
- **Several pass but joint support <20:** P4-P3 remains blocked. Expand Bank W families prospectively or revise the joint-support design before any intervention.
- **No model besides Qwen passes baseline flatness:** working-set comparison becomes Qwen-only; OLMo lineage uses its broader development analyses, not this primary.

## 8.4 Intervention producer review

Before Phase 4 freeze, independently review the future Bank-W intervention producer for:

- exact primary cell selection: derived, once;
- low/high load pairing by family and seed;
- candidate-set answer scoring;
- J and matched-control rank/energy equivalence;
- stable seeds shared across models at family level;
- no model selection from intervention outcomes;
- max-T sign-flip implementation using a shared family sign;
- capability-set manifest binding;
- no hidden item dropping after baseline gates;
- resume and per-row immutability.

No Bank-W intervention outcome should run in the side track before the Phase 4 freeze.

---

# 9. Freeze-candidate assembly

## 9.1 Final primary family must be explicit

Before independent review, write one table with no placeholders:

| ID | Hypothesis | Model population | Endpoint | Alternative | Families | SESOI | Power | Status |
|---|---|---|---|---|---:|---:|---:|---|
| P4-P1 or P4-E1 | bridge-specific substitution | Qwen | exact formula | positive or estimation | fixed | fixed | fixed / N/A | primary or estimation |
| P4-P2 | mode by common phase | Qwen | exact interaction | positive | fixed | fixed | >=0.80 | primary or removed |
| P4-P3 | load engagement | eligible set | shared-family max-T | positive | fixed | 0.1585 nat | >=0.80 | primary |

Holm applies only to rows that remain binary primaries.

## 9.2 Environment lock

Freeze:

- exact repo commit;
- exact model revisions and snapshot manifests;
- exact tokenizers and chat templates;
- exact lens hashes and canonical-lens branch rationale;
- CUDA driver, PyTorch, Transformers, Triton, FLA, safetensors, `jlens` versions;
- exact bank, partition, alias, source-verification, and capability manifests;
- exact seeds and RNG algorithms;
- exact parser version;
- exact primary analysis source hashes;
- exact output roots and logical URI map.

## 9.3 Untouched-data audit

An independent reviewer should verify:

- no confirmatory/replication intervention result exists in any registry or run root;
- no side-track job opened Phase 4 untouched families;
- Bank B and Bank W family IDs are absent from development outcome files except where explicitly assigned to development;
- Phase 4 code does not infer side membership from filenames in a way that leaks it to development logic;
- all partitions are hash-pinned and entity/family disjoint;
- no pre-outcome threshold was edited after a target result appeared.

## 9.4 Whole-registry verification

Run a verifier over every live Phase 4 event:

- JSON schema;
- unique event IDs;
- valid supersession graph;
- output existence, size, and hash;
- input hash resolution;
- source commit reachability;
- no output path reused by two live events;
- no missing logical URI;
- no registered file under a temporary/recovery path;
- all figures regenerate from registered tables;
- all analysis envelopes independently reconstruct.

Produce:

```text
manifests/phase4_pre_freeze_inventory.json
manifests/phase4_pre_freeze_inventory.md
```

## 9.5 Review packet

The freeze candidate should include:

```text
SCIENTIFIC_PREREGISTRATION_PHASE4_CANDIDATE.md
FREEZE_GATE_LEDGER_PHASE4.md
PHASE4_DEVELOPMENT_REPORT.md
PHASE4_METHODS_DECISION_RECORD.md
phase4_pre_freeze_inventory.json
PAPER_CONCLUSION_SKELETON.md
parallel_import_inventory.md
```

Stop for independent review and PI sign-off. Do not interpret silence as approval.

---

# 10. Code and test review priorities

## 10.1 Recovery and fit tests

Add or retain tests for:

- local/Drive recovery pair consistency;
- selecting the highest compatible boundary;
- refusing a higher but incompatible checkpoint;
- refusing changed corpus, runtime, model, `jlens`, or fitter hash;
- milestone order 120→250→500→1000;
- exact checkpoint size and tensor inventory;
- no partial milestone registration;
- stable logical URI after local rematerialization.

## 10.2 Selection-margin tests

Synthetic cases should cover:

- identical subspace with swapped duplicate rows;
- near-tied kth/k+1 scores;
- high Jaccard but low projector overlap;
- low Jaccard but projector identity;
- rank-deficient duplicate rows;
- protected rows just outside top-k;
- stable core plus unstable fringe;
- alias-equivalent row swaps.

## 10.3 P4-P2 phase tests

- official on/off prefill state initialization;
- no generated reasoning cell for off mode;
- expected-phase hook fires and wrong-phase refusal;
- cache/no-cache parity at phase boundaries;
- parser behavior on EOS, length, and malformed delimiters;
- accepted answer only after reasoning closure;
- prefill-only intervention cannot leak into final phase via a persistent hook unless intended;
- final-answer intervention starts on the exact first final token;
- deterministic rerun.

## 10.4 Bank-W tests

- exact answer candidate set and tokenization;
- load pair prompt-length equality within seed;
- family and seed balance;
- no answer leakage;
- shared-family signs across models;
- max-T calibration under null and planted alternatives;
- capability model-set binding;
- joint support cannot be repaired by dropping a model.

## 10.5 Bank-B tests

- bridge/answer orthogonalization numerical stability;
- zero answer-span overlap after projection;
- retained bridge norm floor;
- dose matching;
- unrelated and random controls;
- two-component IUT p-value is `max`, not `min`;
- estimation-only mode cannot emit a reject verdict.

---

# 11. Recommended 24-hour schedule after the current VM

The actual A1000 continuation may consume most of one fresh block. Use this order.

## GPU lane

### Block P4.3-A

1. Bootstrap and recover highest A1000 checkpoint.
2. Finish and register A1000.
3. Run A500-to-A1000 structural convergence.
4. Run prompt-323 influence structural audit.
5. Run A500/A1000/published functional gate.
6. Run selection-margin audit on frozen rows.
7. Apply the Q-L1 through Q-L5 branch table.
8. If a canonical instrument is licensed, review and run the consumed-family P4-P2 variance pilot.

### Block P4.3-B, if needed

1. Complete any P4-P2 pilot cell interrupted after a registered row boundary.
2. Run a consumed-family Bank-B orthogonalized-estimand feasibility grid only if its protocol was committed first.
3. No untouched intervention outcome.

## CPU lane

1. Prepare all A1000 successor configs before the hash-binding commit.
2. Implement selection-margin analysis and tests.
3. Review P4-P2 producer.
4. Prepare exact power simulation code for the pilot variance.
5. Draft Bank B decision memo.
6. Build import validators for OLMo and Gemma bundles.
7. Run registry inventory and figure reconstruction.
8. Update conclusion skeleton by branches, not by optimistic prose.

## Never-drop items

1. A1000 safe completion and registration.
2. A500-to-A1000 functional gate.
3. Selection-margin audit.
4. Explicit canonical-lens decision.
5. Freeze checklist update.

## Drop order under compute pressure

```text
Bank-B orthogonalized dev grid
-> prompt-323 full causal rerun if structural effect is tiny
-> P4-P2 pilot completion
-> extra descriptive figures
```

Do not drop the functional gate to fit more structural plots.

---

# 12. Decision ledger for the paper

At the end of Phase 4.3, update each candidate conclusion mechanically.

## C1. Direction content, not dose, carries a causal tail

Already licensed by Phase 3 confirmatory and replication. Phase 4.3 cannot weaken it unless the release audit uncovers a new defect. Lens-fit sensitivity in a new Phase 4 Qwen instrument does not retroactively alter the frozen published-lens Phase 3 result.

## C2. Training changes what occupies or uses the OLMo channel

Development support is strong. It can be upgraded by the OLMo side track’s Bank W and geometry/receiver results, but not by simply adding more narrative to the existing trajectory.

## C3. Qwen uses a bridge-consumable route

P3-P3 supports true-versus-distractor rescue. The semantic swap is development evidence. If Bank B becomes estimation-only, the paper must say that bridge specificity beyond direct answer steering remains estimated rather than confirmed.

## C4. Think training supports external-state substitution

Bank W decides. A positive load-by-derivation/redundancy pattern may upgrade the claim. A null should replace it with the narrower observed trajectory statement.

## C5. The J-lens transport and sparse-coordinate premise is checkpoint- and fit-dependent

The A1000 outcome can strengthen this into a methods result:

- full convergence supports an empirically validated Qwen instrument;
- row-only instability supports dictionary redundancy;
- span or causal instability supports lens-random-effect or consensus methods;
- failure supports a hard applicability warning.

The Gemma side track separately tests finite transport curvature.

---

# 13. Handoff from the three parallel tracks into Phase 5

Phase 5 should not begin merely because three VMs finish. It begins after Phase 4 is frozen or explicitly re-scoped, and after the side-track bundles are reviewed.

Use this router:

```text
OLMo Bank W positive and receiver localized
    -> Phase 5A: training-installed external-state substitution mechanism

Qwen bridge-specific estimate positive and answer-direction-separated
    -> Phase 5B: receiver/path completion on Qwen

Gemma exact JVP shows finite curvature with localized cause
    -> Phase 5C: architecture-specific nonlinear transport methods

Gemma prompt-specific JVP passes but mean map fails
    -> Phase 5D: context atlas / mixture-of-Jacobians

Qwen A1000 still sparse-causal unstable
    -> Phase 5E: lens-random-effects and consensus-subspace methods

No strong mechanism branch, but Phase 4 primaries freeze cleanly
    -> run Phase 4 confirmatory/replication once, then write the paper
```

The strongest next phase is the branch that converts one current descriptive or causal result into a mechanism with a successful falsifier. Do not automatically pick the branch with the largest quantity of artifacts.

---

# 14. Concrete deliverables

The coding agent should leave the branch with:

```text
interpretability/jspaces/phases/phase4/reviews/jspace_lab_nextsteps_4_3.md
interpretability/jspaces/phases/phase4/reports/INPROGRESS_VM13_<date>.md
interpretability/jspaces/phases/phase4/reports/PHASE4_DEVELOPMENT_REPORT.md
interpretability/jspaces/phases/phase4/preregistration/SCIENTIFIC_PREREGISTRATION_PHASE4_CANDIDATE.md
interpretability/jspaces/phases/phase4/preregistration/FREEZE_GATE_LEDGER_PHASE4.md
interpretability/jspaces/phases/phase4/manifests/phase4_pre_freeze_inventory.json
interpretability/jspaces/phases/phase4/manifests/phase4_pre_freeze_inventory.md
interpretability/jspaces/phases/phase4/paper/PHASE4_METHODS_DECISION_RECORD.md
```

New code, conditional on execution:

```text
jspace_phase4/experiments/p4_qwen_lens_influence_prompt323.py
jspace_phase4/experiments/p4_qwen_selection_margin.py
jspace_phase4/experiments/p4_qwen_mode_variance_pilot.py
jspace_phase4/experiments/p4_parallel_import.py
jspace_phase4/experiments/p4_bank_b_orthogonal_feasibility.py
```

Every producer requires:

- a config file;
- tests;
- immutable per-item or per-layer rows;
- input manifest;
- result envelope;
- registered outputs;
- a figure only when it answers a defined question;
- independent reconstruction where practical.

---

# 15. Completion criteria for Phase 4.3

Phase 4.3 is complete when:

- [ ] A1000 is registered, hash-verified, and recoverable;
- [ ] A500-to-A1000 structural comparison is registered;
- [ ] A500/A1000/published functional comparison is registered;
- [ ] selection-margin audit distinguishes row ties from subspace instability;
- [ ] one Q-L branch is recorded and the canonical-lens decision is explicit;
- [ ] P4-P2 variance/power path is either freeze-ready or removed from the binary primary family;
- [ ] P4-P1 is either prospectively repaired with adequate power or reclassified estimation-only;
- [ ] OLMo Bank-W capability events are imported and the eligible model set is fixed;
- [ ] final primary family, Holm count, SESOIs, family counts, and power are filled;
- [ ] entire Phase 4 registry and live outputs verify;
- [ ] no untouched intervention outcome exists;
- [ ] independent-review packet is ready;
- [ ] PI, freeze commit, and freeze tag remain unsigned until the user actually signs.

If all scientific and mechanical gates pass, the next document is the Phase 4 freeze record and execution sheet, not another broad development plan. If one primary remains infeasible, narrow the primary family honestly and freeze the study that can actually answer its questions.
