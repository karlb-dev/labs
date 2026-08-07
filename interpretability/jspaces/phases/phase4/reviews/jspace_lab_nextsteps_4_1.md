# jspace_lab_nextsteps_4_1.md

## J-space Phase 4: routing mechanisms, training trajectory, and transport validity

**Review target:** `karlb-dev/labs`, branch `interp_jspace_part2`, including the full Phase 3 package, frozen preregistration, primary and replication grids, locked analyses, span audits, prose controls, bridge-mediation work, N8 work, OLMo base-lens fit, the current Phase 3 report and handout, the live VM progress ledger, and the Gemma 4 nonlinear-transport case study.

**Phase decision:** Phase 3 has already crossed its scientific freeze, completed its confirmatory and replication blocks, and discharged the originally defined N8 ladder. The next scientific campaign should therefore be **Phase 4**, not an indefinitely expanding Phase 3. Before new Phase 4 claims launch, perform the short but mandatory Phase 3 release audit in Part I of this document. That audit protects the paper; it does not reopen or rewrite the frozen Phase 3 hypotheses.

**Recommended new namespace:**

```text
interpretability/jspaces/phases/phase4/
run root: special-lab-1/phase4_<date>/
branch: interp_jspace_phase4, cut from the Phase 3 completion tag
```

The already completed OLMo base lens and the post-freeze Phase 3 development results can enter Phase 4 as versioned development inputs. They do not need to be thrown away or dishonestly relabeled as if they had not been seen.

> **Paste-line for the coding/research agent**
>
> Read this file after `jspace_lab_nextsteps_3_1.md`, its accepted addendum, `REPORT_PHASE3.md`, the Phase 3 handout, the frozen Phase 3 preregistration and partition, `inprogress.md`, and the Gemma nonlinear-Jacobian handout. Treat Phase 2 and the frozen Phase 3 confirmatory/replication artifacts as immutable. First execute the Phase 3 release audit in Part I, registering corrections or clarifications without overwriting historical evidence. Do not launch a Phase 4 confirmatory cell until the new Phase 4 package, stable seed contract, scoring contract, item manifests, development gates, power analysis, and preregistration are frozen. Prioritize the OLMo lineage and Qwen bridge/mode mechanisms. Run Gemma only after the OLMo/Qwen core is banked, and frame Gemma as a bounded transport-method autopsy unless an exact-JVP gate licenses more. Commit and push at every boundary, checkpoint GPU work at intervals that lose no more than ten minutes, and keep every number downstream of immutable per-item records.

---

# 0. Executive verdict

## 0.1 Phase 3 is scientifically complete

Phase 3 achieved the job it was created to do:

1. It replaced label-only output protection with **span-safe protection** and showed that the J-specific heavy tail survives the leak-proof instrument.
2. It confirmed that result on Qwen and reproduced it on held-out relation families.
3. It confirmed a true-versus-distractor bridge-protection rescue on Qwen.
4. It replaced the thin released two-hop comparison with a thick, paired direct/composed bank.
5. It showed that Qwen's effect is driven by parametric composition rather than the synthetic Bank S working-memory set, forcing the more precise phrase **knowledge-access channel** rather than **global working-memory workspace**.
6. It independently reproduced the Phase 2 analysis layer and frozen Phase 2 model cells through the N8 ladder.
7. It found a sharp Qwen-versus-OLMo mechanistic split: Qwen appears to consume readable bridge content; OLMo Think does not show the same bridge mediation under the current development assay.
8. It completed the exact matched-control prose study and showed that no tested model earns a clean selectivity claim.

The frozen Phase 3 primary family landed as follows:

| result | frozen estimate | status |
|---|---:|---|
| P3-P2, Qwen span-safe J-specific tail | +0.096 tail-rate excess | rejects and replicates at +0.102 |
| P3-P3, Qwen true-vs-distractor bridge protection | +0.431 nats | rejects |
| P3-P1, Qwen minus OLMo thick composition contrast | -0.261 nats | same-direction estimate, p = 0.062 under disclosed power limits |
| Think vs Instruct thick contrast | +0.01 nats | indistinguishable within a wide interval |
| Qwen Bank S composition term | about +0.02 nats | no working-memory effect identified |

That is a coherent Phase 3 result set, not a pre-freeze pilot waiting for one more model.

## 0.2 Phase 4 should answer a different class of questions

Phase 3 established that the cleaned-up phenomenon exists. Phase 4 should explain **how it is routed, where training changes it, when load engages it, and when the J-lens is a valid transport instrument**.

The most promising Phase 4 thesis is:

> Open models do not share one universal verbalizable workspace geometry. Qwen routes some composed parametric knowledge through a causally readable bridge channel. The tested OLMo checkpoints are organized more by accessibility and output adjacency, with little evidence that their composed answers consume the same bridge representation. Training regime, inference mode, and architecture determine what occupies the verbalizable channel. The J-lens can reveal that organization only where its finite-dose transport premise passes.

This is stronger and more specific than trying to force every result into a single “workspace present versus absent” binary.

## 0.3 Five release issues must be closed before Phase 4 science starts

The latest review found five matters that are important enough to resolve before the Phase 3 paper numbers become the foundation of another campaign:

1. The frozen preregistration says P3-P2 uses a protected-answer stratum, but the locked code tests all Qwen items and the primary grid did not record clean answer rank.
2. Phase 3 primary-grid control seeds depend on Python's process-randomized `hash(item_id)`, so exact control recreation is not guaranteed from the declared seed.
3. The published P3-P1 “wild-cluster 95% CI” is actually `estimate ± 1.96 * family-SE`; the primary randomization p-value is 0.062 while that normal interval barely excludes zero.
4. The advertised N8 Levels 2 and 3 rerun the frozen **Phase 2 N6** cells, not the Phase 3 primary grid. They are valuable, but their scope should be named correctly.
5. The bridge swap experiment scores collapse of the original answer, but does not yet show that probability or generation moves toward the counterfactual answer. It therefore supports a strong routing-disruption claim, not yet a clean semantic-swap claim.

These are repairable. None requires deleting the Phase 3 result. Several are exactly the kind of transparent correction that makes the eventual paper more credible.

## 0.4 Priority order

The recommended order is:

```text
P4-0  Phase 3 release audit and completion tag
P4-1  OLMo lineage localization: base -> 3.0 Think -> 3.1 Think/Instruct
P4-2  Qwen bridge mechanism replication and counterfactual-target test
P4-3  Qwen official thinking-on/off and phase factorial
P4-4  Bank W controlled load study on OLMo pair + Qwen
P4-5  Symmetric lens-fit and capacity study
P4-6  Downstream receiver/localization study
P4-7  Bounded Gemma exact-transport autopsy
P4-8  manuscript, artifact release, and independent Phase 4 reproduction
```

Do not start with Gemma. Gemma is scientifically valuable, but Qwen and OLMo already contain a live causal contrast that can be turned into a mechanism paper. Gemma becomes the methods boundary and architecture appendix after that spine is secure.

---

# Part I. Phase 3 release audit

# 1. Preserve the freeze, then clarify it

## 1.1 Add an immutable freeze record

The tag and evidence event show that Phase 3 froze correctly, but the frozen preregistration file still literally says “CANDIDATE - NOT FROZEN,” and its checklist remains unchecked. The freeze program renamed the file without rewriting its internal status text.

Do not edit the historical tag. Add:

```text
interpretability/jspaces/phases/phase3/preregistration/PHASE3_FREEZE_RECORD.md
```

It should contain:

- freeze commit and tag;
- parent commit and derived seed 85670;
- partition artifact hash;
- 36/36 family counts and intersection counts;
- P3-P3 model decision;
- the exact gate event ID;
- a statement that the stale candidate header is a document-generation defect, not evidence that the freeze failed;
- a hash of the frozen preregistration file;
- links to the frozen tag and current clarification.

Also add a one-paragraph pointer to the top of the **current** report and README. The frozen file remains unchanged.

## 1.2 Add a completion tag only after the audit

After Sections 2 through 7 below are banked, tag:

```text
jspace-phase3-complete-v1
```

Cut the Phase 4 branch from that tag. This creates a clean scientific boundary:

- `jspace-phase3-freeze-v1`: design frozen before primary outcomes;
- `jspace-phase3-complete-v1`: confirmatory, replication, reproduction, and release audit complete.

---

# 2. P0 audit: P3-P2 did not apply the preregistered protected-answer stratum

## 2.1 The mismatch

The frozen preregistration defines P3-P2 on items whose clean answer is inside `protect_k`. The locked analysis currently does this:

```python
q = eff[eff.model == "qwen36-27b"].copy()
q["delta_J"], q["delta_C"] = q.J_eff, q.C_eff
p2 = within_item_label_exchange_tail(q, threshold=-1.0, ...)
```

There is no clean-rank filter. The Phase 3 primary grid also stores no `clean_first_rank` field, so the intended stratum cannot be reconstructed from the existing parquet alone.

This does not mean the +0.096 estimate is false. It means it is an **all-items span-safe specificity test**, not the exact primary population written in the preregistration.

## 2.2 Required repair

Write:

```text
jspace_phase3/experiments/p3_protected_answer_audit.py
```

The script should perform a baseline-only measurement on the frozen confirmatory and replication Qwen items:

1. Load the exact frozen model revision, tokenizer convention, bank versions, partition, and first alias used by the original Phase 3 grid.
2. Recompute only clean logits. Do not rerun or inspect intervention outcomes during rank collection.
3. Record:
   - rank of the first token of the exact scored alias;
   - minimum first-token rank over the frozen accepted alias set;
   - whether each rank is `<= protect_k`;
   - clean probability mass on the accepted alias set;
   - prompt and alias token hashes.
4. Join those ranks to the immutable Phase 3 outcome parquets.
5. Recompute P3-P2 on:
   - the exact scored-alias protected stratum, primary protocol-conformance view;
   - minimum-rank-over-aliases stratum, sensitivity;
   - all items, the originally published estimate.
6. Repeat on the held-out replication partition.
7. Run the threshold curve at -0.5, -1.0, -1.5, and -2.0 nats.
8. Use a plus-one Monte Carlo p-value, never literal `p=0`.

Register:

```text
p3-protocol-audit-protected-answer-qwen-v1
```

## 2.3 Decision rule

- If the protected-stratum estimate remains positive and its randomization interval remains clear, the paper may say the Phase 3 all-items result is reproduced under the preregistered protected-answer subset in a post-freeze conformance audit.
- If it shrinks materially or becomes unresolved, the paper must narrow P3-P2 to the all-items span-safe specificity result and label the protected-answer version unresolved.
- Do not retroactively call the audit preregistered confirmation. Call it what it is: a deterministic protocol-conformance correction using frozen outcomes and newly measured baseline metadata.

## 2.4 Tests

Add tests that fail when:

- an analysis claims a protected-answer stratum without a rank field;
- `protect_k` differs from the frozen config;
- prompt or alias hashes differ from the Phase 3 records;
- a rank measurement accidentally reads an intervention column;
- the confirmatory and replication item sets overlap.

---

# 3. P0 audit: unstable per-item control seeds

## 3.1 The defect

The Phase 3 primary runner derives both condition order and matched-control seeds from Python's built-in hash:

```python
abs(hash(item_id))
```

Python hashes are process-randomized unless `PYTHONHASHSEED` is explicitly fixed. The Phase 3 reproduction bootstrap does not set it. Condition order should not affect a stateless teacher-forced model, but the matched control is a genuinely random subspace, so its realization can change.

## 3.2 Stable seed contract for all future work

Create one shared helper:

```python
from __future__ import annotations

import hashlib


def stable_seed(namespace: str, item_id: str, base_seed: int = 0) -> int:
    payload = f"{namespace}\0{item_id}\0{base_seed}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)
```

No Phase 4 producer may call `hash()` for a scientific seed. Add a repository test that searches the Phase 4 package AST for `hash(...)` inside RNG construction.

## 3.3 Phase 3 control-seed sensitivity audit

Write:

```text
jspace_phase3/experiments/p3_control_seed_audit.py
```

Run on a balanced frozen subset first, then expand to full Qwen if the subset shows meaningful variation:

- 40 confirmatory and 40 replication items;
- at least 12 canonical families;
- seeds 11, 101, 1009, 4242, and 31337;
- same span-safe J arm, each seed generating a new exact rank-and-energy matched control;
- J arm and baseline must reproduce exactly across seed runs;
- control effects, tail membership, P3-P2 specific effect, and P3-P1 within-fact effects are compared across seeds.

Report:

- control mean and tail rate by seed;
- pairwise correlation of control deltas;
- specific-effect estimate by seed;
- worst movement of P3-P2 and P3-P1;
- seed-ensemble mean and interval;
- whether any published decision changes.

Register the historical limitation explicitly even if the result is robust:

```text
p3-control-seed-contract-audit-v1
```

## 3.4 Gate

The Phase 3 completion tag requires one of these outcomes:

- **ROBUST:** every seed preserves the qualitative and inferential conclusion;
- **SEED-SENSITIVE BUT BOUNDED:** point estimates move, but the ensemble supports the same narrow claim;
- **DECISION-SENSITIVE:** the paper must downgrade or replace the affected Phase 3 claim.

Do not try to infer the historical Python hash salt. It was not recorded. Test the scientific robustness instead.

---

# 4. P0 audit: P3-P1 interval and p-value use different inferential objects

## 4.1 What the code currently does

The primary P3-P1 test is a family sign-flip randomization test. With 17 families it reports approximately `p=0.062`.

The value called `ci_wild_cluster` is not a confidence interval. `wild_cluster_bootstrap_t` returns an estimate, standard error, t statistic, and p-value. The figure then constructs:

```python
estimate - 1.96 * se, estimate + 1.96 * se
```

That normal interval barely excludes zero, while the primary randomization p-value does not reject. Different valid procedures can disagree, but the current label “wild-cluster 95% CI” is inaccurate and creates a visual contradiction.

## 4.2 Required closeout analysis

Write:

```text
jspace_phase3/experiments/p3_inference_audit.py
```

For P3-P1:

1. Enumerate all `2^17 = 131,072` family sign patterns exactly. There is no reason to Monte Carlo this case.
2. Report the exact randomization p-value.
3. Invert the shifted sign-flip test over a dense effect grid to obtain a randomization-compatible confidence set for the family-weighted effect.
4. Compute a proper wild-cluster percentile-t interval, storing the bootstrap distribution or its deterministic hash.
5. Retain the normal `estimate ± 1.96*SE` interval but label it **normal small-cluster approximation**.
6. Report family leave-one-out, relation-group weighting, item weighting, and median-of-family-means.
7. Do not change the frozen rejection decision. The primary p-value remains the declared decision ruler.

For P3-P2 and P3-P3:

- replace zero Monte Carlo p-values with `(extreme + 1)/(draws + 1)`;
- report the number of randomization draws;
- derive randomization-compatible intervals or report effect-size bootstrap intervals separately and name them precisely.

## 4.3 Paper wording

The clean sentence is:

> P3-P1 was directionally consistent on both partitions but did not cross the preregistered randomization-test threshold. Normal and bootstrap intervals are reported as estimation summaries, not as a second decision rule.

Do not call it a “CI-clean near miss.” The evidence is useful precisely because the project can leave a strong-looking estimate unresolved without squeezing it through a second ruler.

---

# 5. P0 audit: distinguish Phase 2 reproduction from Phase 3 reproduction

## 5.1 Current scope

The existing N8 Level 2 and Level 3 runner relaunches:

```text
jspace_part2.experiments.confirmatory_protected_grid
```

It therefore reproduces Phase 2 N6 sentinels and one full Phase 2 Qwen cell. This is excellent evidence for the inherited assay, but it is not a clean-room reproduction of the Phase 3 span-safe primary grid or Phase 3 locked analysis.

## 5.2 Rename in documentation

In the current report, handout, and release manifest, call the existing ladder:

```text
N8-P2-L1
N8-P2-L2
N8-P2-L3
```

Add a short clarification event. Do not alter the old evidence IDs.

## 5.3 Add a real Phase 3 ladder

### N8-P3-L1: analysis reproduction

A narrative-blind process receives only:

- frozen Phase 3 parquets;
- partition artifact;
- analysis schema;
- commands and tolerances;
- no expected values, report, figures, or registry prose.

It must regenerate:

- P3-P1 exact and declared estimands;
- P3-P2 all-items and protected-stratum audit views;
- P3-P3;
- Holm adjustment;
- estimation targets;
- replication analysis.

### N8-P3-L2: sentinel reproduction

Rerun at least 20 Phase 3 items per primary model:

- baseline;
- span-safe J;
- label-protected J;
- exact matched control under an explicitly stable seed;
- Qwen bridge arms on composed items.

Baseline and J-arm values should be exact or within a declared numerical tolerance. The historical matched-control row cannot be promised bit-exact because the original hash salt was not recorded; compare its seed-ensemble distribution honestly.

### N8-P3-L3: full cell

Rerun the full Phase 3 Qwen cell under the repaired stable-seed runner. Required comparisons:

- exact baseline and J-arm agreement;
- matched-control distribution and P3-P2 robustness;
- bridge effect and geometry diagnostics;
- Phase 3 locked-analysis estimates under repaired metadata.

Register these as methods evidence and use them as the actual Phase 3 public-release gate.

---

# 6. P1 audit: bridge protection and counterfactual substitution

## 6.1 The present result is important but not yet fully mechanized

The current evidence is striking:

- true bridge protection rescues Qwen relative to distractor protection;
- bridge-only deletion harms Qwen;
- counterfactual bridge substitution crushes the original answer;
- unrelated-content deletion is near zero;
- the same manipulations are weak or inverted on OLMo Think.

However, two reviewer questions remain live.

### Question A: are true and distractor protection geometrically matched?

The runner appends every tokenizer piece for each entity to the protection set. True and distractor entities may differ in:

- number of token IDs;
- effective protected-span rank;
- overlap with the existing clean top-10 span;
- overlap with selected J rows;
- amount of J rank lost after span-safe residualization;
- answer-direction overlap.

A rescue contrast can therefore combine semantic identity with unequal extra protection.

### Question B: did the swap move the model toward the counterfactual answer?

The current mediation script scores the original answer under every arm. A `-4.05` nat drop proves strong disruption. It does not, by itself, prove that the model adopted the counterfactual bridge or preferred the counterfactual answer.

## 6.2 Required geometry audit

For every true/distractor pair, log per layer and position:

```text
n_piece_ids
protected_rank_before
protected_rank_after
added_rank
trace(P_added P_selected)
selected_rank_before_and_after
removed_energy
lost_J_rank
answer_direction_survival
bridge_direction_survival
```

Produce:

- paired distributions true versus distractor;
- a geometry-only prediction of the rescue contrast;
- a subset exactly matched on piece count and added rank;
- a residualized semantic contrast after geometry matching.

## 6.3 Required semantic swap endpoint

For every counterfactual substitution item, score:

```text
lp(original answer)
lp(counterfactual answer)
preference = lp(counterfactual) - lp(original)
greedy original hit
greedy counterfactual hit
other/invalid generation
```

The primary swap statistic should be the within-item change in counterfactual-versus-original preference, not merely loss of the original answer.

Controls:

- true bridge re-injection;
- unrelated entity injection matched for token count, direction norm, answer overlap, and removed energy;
- random direction in the bridge subspace's orthogonal complement;
- counterfactual answer direction injection without bridge substitution;
- at least two counterfactual bridges per fact when the relation permits it.

## 6.4 Replicate P3-P3

The Phase 3 replication configs disable bridge arms. Therefore P3-P3 has no untouched-family replication.

Phase 4 should include a new frozen family split or an untouched extension bank and replicate:

- true-versus-distractor protection rescue;
- bridge-only lesion;
- counterfactual-target preference shift.

Until that lands, the paper should distinguish:

- **confirmatory:** true bridge protection rescues relative to the chosen distractor on the Phase 3 confirmatory partition;
- **development:** the broader Qwen-versus-Think bridge-consumption double dissociation and counterfactual substitution story.

---

# 7. P1 audit: scoring, cohort selection, metadata, and provenance

## 7.1 First-alias-only scoring

The Phase 3 primary grid scores `accepted_answers[0]`, although the shared scoring layer supports alias sets. The direct and composed variants share the answer, so the within-fact contrast is reasonably protected, but cross-model tokenization and surface-form sensitivity remain.

Run a frozen-item sensitivity subset with:

- first alias, historical result;
- canonical alias;
- logsumexp over a prefix-disjoint accepted alias set;
- max alias, diagnostic only.

Phase 4 prospective primary scoring should use logsumexp over a frozen, audited, prefix-disjoint alias set, with canonical and first-alias sensitivities.

## 7.2 Boundary-safe generation grading

The offline G5 regrade checks normalized substring containment. Require token or word boundaries so a short alias cannot match a longer unrelated word.

A safe normalized sequence matcher is:

```python
def contains_alias(generated: str, alias: str, normalize) -> bool:
    g = normalize(generated).split()
    a = normalize(alias).split()
    return any(g[i:i + len(a)] == a for i in range(len(g) - len(a) + 1))
```

Special cases such as punctuation-free chemical symbols and numbers should have explicit, tested policies.

Hand-audit at least 100 positives and 100 negatives, stratified by model, family, and alias length. Store the worksheet.

## 7.3 Cohort-selection sensitivity

Phase 3 requires the model to generate an accepted answer for both direct and composed variants before the fact enters the model cohort. This is pre-intervention and defensible, but it conditions on model behavior and may attenuate or distort a cross-model composition contrast.

Report Phase 3 and Phase 4 results under four predeclared populations:

1. strict both-variant generation-capable intersection;
2. direct-capable intersection;
3. answer-preference-capable, where the correct answer beats its counterfactual by a frozen baseline margin;
4. all source-verified facts with baseline ability as a covariate.

The strict cohort may remain primary, but the result should not depend entirely on one inclusion mechanism.

## 7.4 Metadata correction

`p3-replication-analysis-v1` is registered with the confirmatory tier because the analysis module uses a constant tier even when `--side replication` is passed.

Add an event-sourced metadata correction. Do not rewrite the old row.

## 7.5 Release manifest

Many Phase 3 registry events carry empty `inputs` despite depending on banks, partitions, configs, model revisions, lenses, and scoring conventions. Build one release manifest containing:

- code commit and tag;
- model hub ID and revision;
- config and tokenizer hashes;
- lens hashes;
- bank hashes;
- partition hash;
- prompt/alias manifest hashes;
- environment and dependency lock hash;
- seed algorithm and seeds;
- raw parquet hashes;
- result-envelope hashes;
- figure producer and source hashes.

Every paper table and figure should cite a live evidence ID and one release-manifest row.

## 7.6 Resume-state contract

Current runners load `state.json` without proving that it belongs to the current config, code, model, banks, or partition. Phase 4 must write a state header and refuse mismatches:

```python
state_header = {
    "code_commit": code_commit,
    "config_sha256": sha256_file(config_path),
    "model_revision": model_revision,
    "tokenizer_sha256": tokenizer_sha,
    "lens_sha256": lens_sha,
    "bank_sha256": bank_sha,
    "partition_sha256": partition_sha,
    "seed_contract": "sha256-v1",
}
```

On resume, compare every key before reading `done` or `rows`.

## 7.7 Final Phase 3 state-of-record report

The living report intentionally preserves history, but the handout now contains stale statements such as “N8 L2-L3 open” after later pages say complete, and an old “54 of 72 families” heading after the bank reaches 72.

Create:

```text
reports/PHASE3_STATE_OF_RECORD.md
reports/handout/jspace_phase3_final.pdf
```

The state-of-record document should contain only:

- frozen design;
- final primary and replication results;
- release-audit corrections;
- exact claim ladder;
- limitations;
- artifact map.

Move the chronology to an audit appendix. This will make the paper much easier to review.

---

# Part II. Phase 4 scientific program

# 8. Phase 4 research question

The campaign now has enough evidence to stop asking only whether a verbalizable causal channel exists. The sharper question is:

> What computation occupies the verbalizable channel in each model, how does post-training reroute that computation, and under what transport geometry can the channel be measured causally?

Phase 4 should separate five candidate organizations:

1. **Bridge-routing channel:** composed answers depend on a readable intermediate entity, as Qwen currently appears to do.
2. **Accessibility channel:** lesion cost depends on how locked-in or rehearsed the answer is, as the OLMo cluster currently suggests.
3. **Working-set channel:** lesion cost rises with controlled in-context load, not merely parametric composition.
4. **Externalized channel:** thinking mode moves state into generated tokens and changes which phase depends on internal J-content.
5. **Transport-invalid regime:** a fixed linear J-map is not licensed, as the Gemma pilot may demonstrate.

The paper becomes compelling if it can map models to these organizations with matched interventions rather than merely placing them on a one-dimensional “workspace strength” ladder.

# 9. Phase 4 governance

## 9.1 Separate package and registry

Create:

```text
interpretability/jspaces/phases/phase4/
    jspace_phase4/
    configs/
    data/
    tests/
    preregistration/
    reports/evidence_events.jsonl
    paper/
    reviews/
```

Allowed tiers:

```text
phase2-confirmatory-import
phase3-confirmatory-import
phase3-replication-import
phase4-development
phase4-confirmatory
phase4-replication
methods
```

The Phase 4 registry must reject creation of new Phase 2 or Phase 3 evidence. Imported rows point to immutable source artifacts.

## 9.2 Development before freeze

Phase 4 begins with development gates for:

- OLMo lineage feasibility;
- Qwen bridge geometry and swap-target endpoint;
- Qwen mode parsing and phase hooks;
- Bank W baseline capability and shortcut audits;
- lens-fit convergence and capacity stability;
- Gemma exact-JVP feasibility.

Only after those gates should the Phase 4 primary family be frozen.

## 9.3 Recommended confirmatory family

Do not freeze all exploratory ideas as primaries. A compact family is more credible:

- **P4-P1: Qwen bridge substitution.** Counterfactual bridge substitution increases counterfactual-versus-original answer preference relative to matched unrelated substitution, on untouched families.
- **P4-P2: Qwen mode-by-phase interaction.** The effect of a span-safe bridge/J lesion on final answer quality differs between official thinking-on and thinking-off modes and between prefill, reasoning-token, and final-answer phases.
- **P4-P3: load engagement.** The span-safe J-specific effect increases with controlled working-set load relative to the exact matched control on at least one preregistered model, with the model-by-load interaction estimated across the OLMo pair and Qwen.

The OLMo lineage trajectory should initially be a named estimation target. Promote one adjacent-checkpoint contrast to the primary family only if the development leg identifies a clear, pre-outcome transition and a new untouched item bank is available.

## 9.4 No post-hoc rescue of P3-P1

Phase 4 may build a higher-power cross-model composition experiment, but it must receive a new hypothesis and new data. It must not be described as “rerunning P3-P1 until significant.” The Phase 3 near miss stays exactly what it was.


---

# 10. Workstream A: OLMo training-trajectory localization

## 10.1 Why OLMo is the highest-priority model family

The OLMo line provides something Qwen cannot: a comparatively legible post-training trajectory on the same architecture and pretraining base. Phase 3 currently contains the 3.1 Think/Instruct pair and has already completed a 120-prompt lens for the base checkpoint. The immediate missing point is OLMo-3-32B-Think, followed by the base model's G5 and span-safe development cell.

This work should answer:

1. Does the span-safe lesion organization already exist in the base model?
2. Does reasoning post-training create, remove, or merely rescale bridge dependence?
3. When does the strong output-adjacent selection pressure appear?
4. Is the accessibility gradient a stable lineage property?
5. Does the unusual positive Bank S composition term on 3.1 Think begin earlier?
6. Does post-training change capacity, channel content, or only downstream consumption?

## 10.2 Immediate continuation from the current state

The base lens is complete and registered. Do not refit it.

Run, in order:

```text
A0  G5 on OLMo-3-32B-Think
A1  span-safe development grid on OLMo-3-32B-Think
A2  G5 on OLMo-3 base
A3  span-safe development grid on OLMo-3 base
A4  common-lens and own-lens cross-checks
A5  four-checkpoint trajectory synthesis
```

Use stable Phase 4 seeds and the repaired Phase 4 scoring contract. The existing Phase 3 frozen families may be used for **development trajectory plots**, because those outcomes are already known. Any binary lineage claim requires a new Phase 4 holdout.

## 10.3 Minimal per-checkpoint battery

For each checkpoint:

- baseline capability on Bank F, Bank S, and later Bank W;
- span-safe J arm;
- exact instantaneous rank-and-energy matched control;
- label-protected J arm for continuity;
- protected-energy-matched leakage control;
- mechanics random;
- logit span-safe and label-protected controls;
- prose guard;
- selected-ID and protected-span geometry logging;
- corrected capacity at L24/L32/L40 or relative-depth equivalents;
- G4 swap positive control;
- bridge-only lesion, true bridge protection, distractor protection, unrelated-content lesion, and counterfactual-target preference on a development subset.

Per checkpoint report:

```text
capacity
output-selection pressure
protected-span overlap
span-safe specific mean and tail
Bank F direct/composed contrast
Bank S direct/composed contrast
accessibility slope
prose standardized cost
bridge rescue
bridge lesion
counterfactual target shift
```

## 10.4 Common-lens versus own-lens design

A trajectory can be confounded if every checkpoint gets a different coordinate system. Use two complementary views.

### View 1: common lens

Choose one frozen OLMo lens, preferably the base or 3.1 Think lens, and apply it to every checkpoint under the recipient model's own unembedding and final norm. The architecture and vocabulary are shared, and earlier transfer work showed the output basis is nearly unchanged.

This view asks:

> Under one fixed coordinate frame, did the model's use of the channel change?

### View 2: own lens

Use each checkpoint's own fitted lens.

This view asks:

> Is the phenomenon robust to fitting the best available local transport frame for each checkpoint?

### Required trajectory plot

For every metric, draw both lines:

```text
common-lens trajectory
own-lens trajectory
```

A scientific conclusion is strongest when both agree. If they disagree, that disagreement is itself the result: the coordinate system changed under post-training.

## 10.5 Dictionary and projector trajectory

Do not compare only behavior. Measure how the J dictionaries move:

- row-wise cosine for shared token IDs;
- centered kernel alignment or linear CKA on sampled rows;
- principal angles between selected spans;
- selected-ID Jaccard per item;
- effective-rank distribution;
- protected-span overlap;
- answer-direction survival;
- bridge-direction survival;
- cross-check on independent fitting corpora where available.

Separate:

```text
representation drift
selection drift
downstream-consumption drift
behavioral effect drift
```

This prevents a vague “post-training reorganized the workspace” conclusion when the actual change may be in only one layer of the assay.

## 10.6 Lineage decision tree

### Outcome A: base and every OLMo checkpoint lack bridge mediation

Then the strongest OLMo claim is:

> OLMo's verbalizable channel is consistently accessibility-organized across pretraining and post-training; the tested reasoning/instruction stages rescale lesion severity but do not create Qwen-like bridge routing.

Stop adding OLMo intermediate checkpoints unless another metric has a clear transition.

### Outcome B: bridge mediation appears between base and 3.0 Think

Then post-training likely creates a readable intermediate route. Fetch the nearest publicly available SFT/DPO intermediate checkpoints and localize the transition.

### Outcome C: bridge mediation appears only in one 3.1 branch

Then prioritize matched SFT/DPO branch points. This is the cleanest evidence that objective-specific post-training changes internal routing.

### Outcome D: common-lens and own-lens trajectories disagree

Then fit/dictionary drift is a central finding. Add a corpus and fit-size study before interpreting behavior.

## 10.7 The Think Bank S anomaly

The current Phase 3 estimate says composed Bank S items lose **less** than direct items under span-safe J on 3.1 Think. Before inventing a theory, run a forensic decomposition:

- direct versus composed prompt length;
- answer position and token count;
- baseline LP and entropy;
- selected rank and removed energy;
- protected span rank;
- number of active bridge/state-token directions;
- direct/composed selected-ID overlap;
- whether composed prompts move the answer into the clean top-k earlier;
- whether the matched control changes with length;
- family leave-one-out;
- alias sensitivity;
- syntax/template effects.

Test three candidate explanations:

1. **Redundancy:** the composed prompt contains more cues, so deleting one J route hurts less.
2. **Externalized state:** the prompt itself stores the state, reducing reliance on the internal channel.
3. **Dose dilution:** a longer sequence spreads the lesion over more positions or changes the selected span.

A new controlled Bank W experiment should decide among them. Do not interpret the positive term before that.

## 10.8 When to add intermediate OLMo checkpoints

Only add SFT/DPO intermediates when the four-point trajectory identifies a transition interval. Otherwise they are expensive decorative dots.

A checkpoint is admitted only if:

- exact revision and training relationship are documented;
- tokenizer and architecture match;
- a lens can be fit or a transfer gate passes;
- G4 passes;
- the same Phase 4 bank and scoring contract apply;
- it narrows an already observed transition.

---

# 11. Workstream B: Qwen bridge channel, replicated and causally completed

## 11.1 The core Qwen question

Phase 3 suggests that Qwen routes composed parametric answers through a readable bridge entity. Phase 4 should turn that suggestion into a closed causal chain:

```text
bridge is represented
-> bridge-specific lesion hurts
-> true bridge protection rescues
-> wrong bridge substitution shifts the model toward the wrong answer
-> the effect replicates on untouched families
-> downstream recipients can be localized
```

At present, the first three links are strong. The fourth and fifth need direct evidence.

## 11.2 Build a fresh bridge-mechanism bank

Author a Phase 4 Bank B with at least:

- 40 canonical relation families;
- 4 to 6 facts per family;
- one true bridge;
- two plausible counterfactual bridges;
- original and counterfactual answers;
- direct, composed, bridge-supplied, and counterfactual-supplied variants;
- source verification;
- no overlap with Phase 2 or Phase 3 facts, bridges, or answers;
- matched token-length strata for true and counterfactual bridges;
- explicit ambiguity notes;
- answer-preference baseline metrics.

Prefer relations where changing the bridge deterministically changes the answer:

```text
country -> currency
person -> birthplace country -> currency
element -> atomic number -> parity/category
author -> nationality -> language
river -> endpoint country -> capital/currency
object -> material -> property
```

Avoid counterfactuals that make the prompt nonsensical.

## 11.3 Geometry-matched bridge protection

For each item, construct distractor/counterfactual bridge protection that matches:

- number of tokenizer pieces;
- effective added protected-span rank;
- overlap with the clean top-k output span;
- overlap with selected J rows;
- baseline activation score distribution;
- bridge-answer direction cosine where feasible.

If exact lexical matching is impossible, use residualized matched subspaces rather than raw token-set size.

Record both semantic and geometric identities. Never label a distractor “matched” without a per-item match report.

## 11.4 Counterfactual substitution endpoint

For each arm, calculate:

```text
orig_lp
cf1_lp
cf2_lp
orig_minus_cf_preference
greedy_orig
greedy_cf1
greedy_cf2
invalid_or_other
```

Primary substitution estimand:

```text
[lp(cf) - lp(orig)]_counterfactual_bridge_injection
- [lp(cf) - lp(orig)]_matched_unrelated_injection
```

Secondary:

- flip rate from original to the intended counterfactual;
- calibration of both answers;
- dose response;
- generated explanation references the substituted bridge;
- recovery when the correct bridge is re-protected after substitution.

This is the difference between “the model was disrupted by a wrong direction” and “the model read and followed the semantic counterfactual.”

## 11.5 Lesion, rescue, substitution, and patching factorial

Recommended arms:

| arm | purpose |
|---|---|
| baseline | clean reference |
| span-safe J | general J-specific lesion |
| exact matched control | dose specificity |
| bridge-only lesion | necessity of true bridge channel |
| matched unrelated lesion | semantic specificity |
| true bridge protected | rescue |
| geometry-matched distractor protected | rescue control |
| true bridge removed + true bridge re-injected | self-rescue positive control |
| true bridge removed + counterfactual bridge injected | substitution |
| true bridge removed + unrelated injected | injection mechanics control |
| answer direction only | downstream answer staging diagnostic |

Calibrate injection dose on a development split without choosing the sign or dose from answer outcomes. Use geometric and activation-scale gates only.

## 11.6 Replication

Freeze Phase 4 Bank B at the family level. P4-P1 uses the confirmatory side, then the exact same analysis runs once on the replication side.

A strong result requires:

- original-answer damage under bridge lesion;
- counterfactual preference increase under counterfactual substitution;
- true bridge protection rescue;
- unrelated controls near zero;
- same signs on held-out families.

## 11.7 Layer, position, and phase localization

Run the mechanism at:

- individual layers and short bands;
- source positions containing the first-hop cue;
- the final prompt position;
- all prompt positions;
- decode steps before and after the bridge becomes explicit;
- prefill-only, decode-only, and both.

The goal is to learn whether Qwen's bridge channel is:

- computed locally near the cue;
- broadcast to the final prompt position;
- held persistently;
- recreated during generation;
- merely read at the answer boundary.

Plot a bridge causal surface:

```text
source layer x position/phase -> lesion, rescue, substitution effect
```

## 11.8 Downstream receiver search

For items with strong bridge mediation:

1. Identify downstream attention heads and MLP blocks whose input/output directions align with the bridge J direction.
2. Rank recipients by gradient-based contribution and activation patching.
3. Lesion the source J direction, then patch the candidate recipient activation from clean to ablated runs.
4. Test whether recipient patching rescues original-answer preference.
5. Ablate the recipient alone and test whether bridge substitution loses its effect.

Controls:

- random recipient at same layer;
- matched high-activation component;
- answer-direction recipient;
- unrelated bridge recipient;
- family holdout.

The paper does not need a complete circuit atlas. One robust source-to-receiver path would materially strengthen the “broadcast knowledge-access channel” interpretation.

---

# 12. Workstream C: Qwen official thinking mode and phase-resolved causality

## 12.1 Why this is a high-value same-weights experiment

The official thinking toggle changes inference behavior without changing model weights. It is therefore the cleanest available test of whether externalized reasoning changes dependence on the internal bridge channel.

Earlier chat-mode teacher-forced bare-answer scoring was invalid because an open reasoning register changes the continuation distribution. Phase 4 must use generation-based endpoints.

## 12.2 Factorial design

Factors:

```text
mode: thinking_on / thinking_off
phase: prefill / reasoning_tokens / final_answer / all
intervention: span_safe_J / exact_matched / bridge_only / unrelated
bank: F / B / W
```

The primary cell should remain small enough to interpret. Recommended primary family:

- Bank B composed items;
- thinking on versus off;
- bridge-only lesion versus matched unrelated lesion;
- prefill-only versus reasoning-only versus final-answer-only.

## 12.3 Phase implementation

Define explicit phase states:

```text
PROMPT_PREFILL
THINK_DECODE
ANSWER_DECODE
INACTIVE
```

Do not infer phase only from token count. Use a robust parser for model-specific reasoning delimiters, including:

- no opening delimiter;
- no closing delimiter;
- multiple delimiters;
- EOS inside thinking;
- answer produced before closure.

Every trace stores:

- token IDs and decoded text;
- phase per token;
- hook fire counts;
- selected directions;
- protected sets;
- bridge rank trajectory;
- answer rank trajectory;
- generation budget and stop reason.

## 12.4 Endpoints

Primary:

- final answer exact/alias accuracy;
- original-versus-counterfactual answer preference from a post-generation scoring pass;
- bridge-substitution flip rate.

Secondary:

- think-block closure rate;
- reasoning token count;
- latency to bridge mention;
- bridge J-rank before textual mention;
- answer J-rank before final answer;
- self-correction after wrong bridge injection;
- generated rationale consistency with the chosen answer.

Do not use “answer appears anywhere in 400 tokens” as a final-performance endpoint.

## 12.5 Controls for externalization

Add generation controls that distinguish useful externalized reasoning from generic extra tokens:

- correct concise rationale supplied in prompt;
- wrong rationale supplied;
- shuffled rationale;
- length-matched neutral filler;
- thinking budget truncated at fixed lengths;
- answer-only mode;
- bridge explicitly supplied.

Interpretation:

- If any extra tokens rescue equally, the mechanism is generic compute/time.
- If correct rationale or true bridge uniquely rescues, externalized semantic state is doing causal work.
- If thinking-on reduces prefill bridge dependence but increases decode dependence, the channel is being reconstructed online.

## 12.6 Primary interaction

A defensible P4-P2 statistic is:

```text
[(bridge lesion - matched unrelated) thinking_on]
- [(bridge lesion - matched unrelated) thinking_off]
```

computed separately for prefill-only and reasoning-only interventions, then combined as a mode-by-phase interaction.

Freeze one direction only after development gates establish that the endpoints are measurable and not dominated by parser failures.

---

# 13. Workstream D: Bank W controlled working-set load

## 13.1 Why Bank W is the decisive “workspace” test

Phase 3's Bank S did not show the Qwen composition effect. That result rules against casually calling the parametric bridge channel a working-memory workspace, but Bank S covers only a limited set of synthetic templates.

A proper load experiment should manipulate the amount of state that must remain available while holding answer type, surface form, and knowledge requirements constant.

## 13.2 Task families

Build at least six independent template superfamilies.

### W1: variable binding

```text
Ava has key R.
Ben has key G.
Cara has key B.
...
Who can open door G?
```

Load is number of bindings plus distractors.

### W2: stateful updates

```text
x = 2
x += 4
y = x - 1
x = y * 2
...
What is x?
```

Load varies active variables and update depth.

### W3: graph/path state

Give a small directed graph and ask for a path-dependent node/property. Load varies nodes, branching, and path length.

### W4: deferred recall

Present labeled facts, insert an unrelated reasoning section, then query one item. Load varies retained items and delay.

### W5: ordered stack or queue

Push/pop or enqueue/dequeue operations with controlled depth. Query the top/front or an intermediate state.

### W6: dual task

Maintain bindings while performing a separate arithmetic or classification subtask. Load varies both storage and interference.

Optional:

- n-back-like token recall;
- reversible mappings;
- order-scrambled question execution;
- multi-entity role assignment.

## 13.3 Load levels

Use a shared scale when feasible:

```text
load = 1, 2, 4, 6, 8
```

Each base item should have nested load variants, so the same underlying answer relation can be compared within item. Randomize entity names, values, and surface forms while preserving a family-level template.

## 13.4 Shortcut and capability gates

Before any intervention:

- baseline accuracy between 40% and 95% at the analyzed loads;
- direct answer not extractable from the last sentence alone;
- shuffled-state control falls to chance;
- query-only control falls to chance;
- distractor-removal improves rather than harms;
- answer token length balanced across loads;
- prompt length sensitivity separated from state load;
- models prefer the correct answer over matched counterfactuals;
- no family contributes more than 10% of a partition.

Create explicit prompt-length controls with equal token count but no additional state.

## 13.5 Intervention family

Per item/load/model:

- baseline;
- span-safe J;
- exact matched control;
- label-protected J, secondary;
- state-token-specific lesion;
- unrelated state-token lesion;
- persistent matched control;
- prefill-only and decode-only where generation is used.

## 13.6 Primary load estimand

For each model:

```text
specific(load) = [J(load) - baseline(load)]
                 - [control(load) - baseline(load)]

load_interaction = specific(high_load) - specific(low_load)
```

Prefer a slope over the full load ladder in the hierarchical sensitivity model, but freeze a simple high-minus-low randomization statistic as the primary.

The model contrast is:

```text
load_interaction_Qwen - mean(load_interaction_OLMoThink,
                             load_interaction_OLMoInstruct)
```

## 13.7 Interpretation rules

- **Positive load interaction, clean controls:** evidence that the J-channel participates in a working-set function.
- **Flat with equivalence:** closes the controlled-load escape hatch for that model and dose.
- **Prompt-length control tracks load:** effect is sequence/dose geometry, not working set.
- **State-specific lesion but no general J effect:** working state exists but is not captured by the top-k verbalizable frame.
- **Qwen parametric effect without Bank W effect:** retain “knowledge-access channel.”
- **OLMo Bank W effect despite weak parametric bridge mediation:** OLMo may use the channel for in-context state rather than stored knowledge.

## 13.8 Replication

Author enough template families for a family-disjoint split. At least 24 confirmatory and 24 replication template families across the six superfamilies is preferable. Never split parameter instantiations of the same template across sides.

---

# 14. Workstream E: symmetric lens-fit and capacity study

## 14.1 The current asymmetry

Qwen's capacity estimate uses a published n=1000 lens, while the OLMo campaign lenses use n=120. The Phase 3 causal results do not disappear because of this difference, but any capacity-versus-routing argument remains vulnerable until fit size and recipe are symmetric.

## 14.2 Qwen nested convergence study

Fit Qwen lenses at:

```text
n = 120, 250, 500, 1000
```

Use nested prompt prefixes under one frozen corpus and recipe. Also fit an independent draw at n=120 and preferably n=500.

Separate two comparisons:

1. **Fit-size convergence:** same corpus and recipe, n varies.
2. **Recipe/corpus transfer:** campaign n=1000 versus the published lens.

## 14.3 Metrics

For every layer and pair of lenses:

- Frobenius norm and merge stability;
- row-wise token-direction cosine;
- CKA on sampled dictionary rows;
- selected-ID Jaccard on frozen items;
- selected-span principal angles;
- protected-span overlap;
- occupancy distribution;
- centered excess capacity;
- G4 swap positive control;
- span-safe specific tail;
- bridge rescue and substitution on a small frozen development set.

## 14.4 Decision criteria

A smaller lens is validated when:

- occupancy changes by at most one median unit;
- centered excess moves by less than one absolute percentage point or a frozen relative bound;
- selected-ID Jaccard is at least 0.7;
- span-safe causal effect moves less than 0.15 nats or a frozen tail-rate margin;
- G4 remains positive;
- bridge mechanism signs are unchanged.

If Qwen n=120 already reproduces the n=1000 picture, fit size is not the capacity explanation. If it does not, the paper must stop comparing Qwen and OLMo capacity without a converged common recipe.

## 14.5 OLMo scaling only as needed

Do not automatically fit 250/500 lenses for every OLMo checkpoint. First run a nested 120/250/500 study on one representative OLMo model, preferably 3.1 Think. Expand only if capacity or causal selections have not stabilized.

## 14.6 Capacity is not a three-point regression

The current OLMo pair and Qwen give too few independent model-family points to test capacity as a moderator. Do not fit a model-level regression and quote a p-value.

Phase 4 may support a moderator analysis through either:

- at least six assay-valid checkpoints spanning two or more size families; or
- within-model local channel-load measures that predict item-level lesion effects.

Until then use “co-varies across the tested models” and explicitly separate family, scale, fit recipe, and post-training.

---

# 15. Workstream F: channel load, persistence, and downstream broadcast

## 15.1 Move beyond one tail-rate number

The Phase 3 tail is real, but a tail-rate alone does not say how a channel is used. Add per-position and per-item channel-state measurements:

- occupancy crossing and local support size;
- selected rank;
- persistent selected directions across layers/positions;
- bridge and answer rank trajectories;
- selection pressure toward protected output rows;
- span-safe lost rank;
- removed energy;
- receiver fan-out;
- time until textual externalization.

## 15.2 Persistent frame versus instantaneous frame

Phase 3 found a small but nonzero effect from a persistent matched control. Phase 4 should treat temporal coherence as a first-class axis:

- independent random subspace per position;
- one random frame per item and layer;
- one random frame per item across the band;
- J-selected persistent frame;
- persistence-selected J directions;
- exact rank/energy matching for every variant.

This tests whether the causal channel is a stable frame or a sequence of rapidly changing token coordinates.

## 15.3 Broadcast assay

For a selected bridge direction at source layer `l`, measure downstream access by:

- Jacobian alignment into attention and MLP inputs;
- activation-gradient product;
- receiver count above a frozen control-calibrated threshold;
- causal patch rescue;
- persistence of the direction in later residuals;
- effect on non-answer tokens and parallel tasks.

Compare Qwen and OLMo on the same bridge bank.

Predictions:

- Qwen bridge directions should have stronger, more specific downstream recipients and patch rescue.
- OLMo selected directions may be more output-adjacent or diffuse, with weaker bridge-specific receivers.

## 15.4 Mediation analysis

Use an intervention-based mediation chain rather than correlational language:

```text
source bridge lesion -> answer preference change
source bridge lesion + recipient patch -> rescue
recipient lesion alone -> answer preference change
counterfactual source injection -> recipient state shift -> counterfactual answer shift
```

A receiver is credible only if all four links behave directionally and matched controls stay small.

---

# 16. Workstream G: bounded Gemma exact-transport autopsy

## 16.1 Do not abandon Gemma, and do not let it swallow the project

Gemma is not a failed model cell. It is a methods-boundary case with a potentially important lesson: a fixed linear transport map may be useful on OLMo and Qwen but not licensed in Gemma's middle band.

The supplied Gemma case study correctly separates:

- differentiability;
- finite-radius local linearity;
- cross-context stability.

The decisive missing validation is an exact autograd JVP against faithful finite differences. Run Gemma only after the OLMo and Qwen core is secure, and cap it at two serious GPU blocks unless a clean mechanism emerges.

## 16.2 G0: exact JVP versus secant

Source layers:

```text
L22, L30, L37, L42, L44, L48, L52
```

Targets:

- residual after each subsequent block;
- final residual before final norm;
- optional pre-softcap and post-softcap logits as separate endpoints.

Directions:

- random tangential to the residual sphere;
- random radial component;
- fitted-J selected directions;
- output-token/logit directions;
- bridge directions from matched factual prompts;
- paired superpositions.

For each prompt, layer, direction, and epsilon:

```text
exact JVP
central finite difference
forward finite difference
delivered perturbation norm ratio
delivered perturbation cosine
tangent cosine
tangent relative error
gain ratio
homogeneity defect
odd-symmetry defect
additivity defect
```

Use fp32 activation injection and log the actual delivered delta after every cast. The model weights may remain in their validated inference dtype, but the overwritten residual and response comparison must avoid the previous bfloat16 floor.

## 16.3 Epsilon ladder

Choose scales relative to the clean residual norm and include:

- the smallest scale with input fidelity above 0.999;
- two intermediate scales;
- the actual Phase 3 ablation-scale range;
- one larger diagnostic scale.

Do not set universal numeric epsilons without measuring delivered fidelity per layer.

## 16.4 Decision logic

### Case A: exact JVP and tiny secant disagree

The harness or hook path is wrong. Stop. Do not interpret curvature.

### Case B: exact JVP matches tiny secant, then error grows with epsilon

Within-context finite-dose curvature is established.

### Case C: prompt-specific JVP works, but corpus-average J fails at tiny epsilon

Mean-Jacobian cancellation or multiple local charts dominate.

### Case D: prompt-specific JVP fails at the smallest faithful scale

Investigate hook placement, source/target mismatch, fused/in-place operations, and state replacement before any architecture claim.

### Case E: late L44-L52 passes while the middle band fails

A late transport regime exists. Test it as output staging, not automatically as a workspace.

## 16.5 Layer-by-layer localization

Inject one source perturbation and measure error after every downstream block. Split each block into:

```text
pre-attention norm
attention output
post-attention norm
pre-MLP norm
MLP gate/value
MLP output
post-MLP norm
residual add
```

Plot incremental error rather than only final error.

The Gemma architecture gives a sharp prediction: if routing concentration causes the failure, error increments should spike disproportionately at the repeated full-attention blocks rather than accumulating uniformly across the five sliding-window blocks.

## 16.6 Mechanism interventions

In priority order:

1. Freeze clean attention weights while allowing value updates.
2. Freeze Q/K routes but leave values live.
3. Freeze values but allow routing changes.
4. Replace each RMSNorm output with its first-order affine expansion around the clean state.
5. Replace the gated MLP locally with its first-order expansion.
6. Freeze only the MLP gate while preserving the value branch.
7. Combine the best two repairs.

A candidate mechanism earns the label only if it reduces tangent error and additivity/homogeneity defects relative to matched OLMo/Qwen controls.

## 16.7 Context heterogeneity

Estimate prompt-specific JVPs along shared probe directions. Report:

- pairwise response cosine across prompts;
- mean-map cancellation ratio;
- within-family versus between-family variance;
- clustering by attention-route signatures, prompt length, and task family;
- performance of one global map versus K context-clustered maps.

If clustered maps rescue small and moderate epsilon while the global map fails, Gemma uses multiple local linear charts. If prompt-specific maps already fail at moderate epsilon, curvature remains the binding limit.

## 16.8 Late-band constructive test

If L44-L52 passes the transport gate, run a small late-band assay:

- token readout;
- capacity and competition;
- persistence;
- lead time before emission;
- span-safe output protection;
- multi-step versus fluency effect;
- exact matched controls.

The key discriminator is lead. A late representation that appears one token before emission is output staging, not the kind of broadcast intermediate channel claimed by the workspace paper.

## 16.9 Nonlinear recovery is secondary

Only after the exact transport diagnosis:

- path-integrated JVP along a fixed intervention path;
- mixture of local linear maps;
- low-rank Hessian-vector correction;
- nonlinear probe for readout.

Do not use a nonlinear probe as causal evidence by itself. A nonlinear causal removal operator lacks the clean geometric controls available to span projection and should be treated as a separate methods project.

## 16.10 Gemma stop rules

Stop the Gemma leg and write the methods result when any of these holds:

- exact JVP/secant validates curvature across middle-band layers and no single sublayer repair explains more than a modest fraction;
- prompt-specific maps fail at intervention scale, making context clustering insufficient;
- late-band success is clearly output staging with negligible lead;
- two full GPU blocks are consumed without a discriminating mechanism.

Continue only when a simple, reproducible cause emerges, such as full-attention routing or gated-MLP curvature, with OLMo/Qwen positive controls.

---

# Part III. Statistics, engineering, and execution

# 17. Phase 4 statistical plan

## 17.1 Paired estimands first

Every causal comparison should reduce to within-item contrasts before aggregation.

For arm `J`, matched control `C`, and baseline `B`:

```text
J_effect = score(J) - score(B)
C_effect = score(C) - score(B)
specific = J_effect - C_effect
```

For direct/composed pairs:

```text
composition = specific(composed) - specific(direct)
```

For counterfactual substitution:

```text
preference = lp(counterfactual) - lp(original)
substitution = preference(wrong_bridge) - preference(unrelated_control)
```

For load:

```text
load_effect = specific(high_load) - specific(low_load)
```

## 17.2 Experimental unit

- relation/template family is the primary cluster;
- fact is paired within family;
- model is paired where the same fact is used;
- prompt paraphrase is not an independent family;
- multiple aliases are scoring variants, not independent samples;
- multiple random seeds are robustness replicates, not new items.

## 17.3 Randomization inference

Use exact enumeration when feasible. For Monte Carlo randomization:

```python
p = (n_extreme + 1) / (n_draws + 1)
```

Store:

- draw count;
- seed;
- statistic definition;
- exact versus Monte Carlo flag;
- randomization-unit description.

## 17.4 Confidence intervals

Name the interval by its actual method:

- family bootstrap percentile interval;
- wild-cluster percentile-t interval;
- randomization-inversion interval;
- normal small-cluster approximation;
- hierarchical posterior/likelihood interval, sensitivity only.

Never call `estimate ± 1.96*SE` a wild-bootstrap interval.

## 17.5 Small cluster safeguards

For every headline:

- exact/randomization p where possible;
- wild-cluster or cluster bootstrap interval;
- leave-one-family-out range;
- relation-group sensitivity;
- item-weighted sensitivity;
- family distribution plot;
- no claim when fewer than three independent clusters exist;
- explicit warning below ten clusters;
- simulation-calibrated type-I error for the exact design.

## 17.6 Heavy tails

Always report:

- mean;
- median;
- tail rate at frozen thresholds;
- conditional tail magnitude;
- ECDF;
- family-level distribution;
- hurdle decomposition;
- threshold curve;
- top influential families.

Do not let a single mean hide the phenomenon again.

## 17.7 Equivalence

A null becomes “closed” only through TOST or an equivalent interval against a frozen SESOI. Failure to reject is inconclusive.

Recommended provisional SESOIs, to be calibrated in development:

```text
answer logprob: 0.15 to 0.30 nats for paired within-fact contrasts
tail rate: 0.10 absolute
accuracy: 0.08 absolute
prose NLL: 0.05 nats/token
counterfactual preference: 0.25 nats
load slope: 0.10 nats per doubling of load
```

## 17.8 Multiplicity

Use separate, small families:

- Family A: three Phase 4 primaries;
- Family B: load subtests;
- Family C: lineage adjacent-checkpoint tests, only if promoted;
- methods diagnostics: descriptive with false-discovery control only where a large screen occurs.

No global BH correction over hundreds of exploratory plots followed by selective storytelling.

## 17.9 Power

Power simulation must preserve:

- family sizes;
- within-fact pairing;
- cross-model pairing;
- zero inflation;
- heavy-tail magnitude;
- correlation among arms;
- cohort-selection rule;
- random control variability;
- family imbalance.

Calibrate the simulator under a true null before using it. The Phase 3 power simulator caught two generator mistakes; keep that self-refuting discipline.

---

# 18. Phase 4 engineering contract

## 18.1 Stable seeds everywhere

No built-in Python hash. No wall-clock seeds. No implicit NumPy global RNG.

Every scientific seed derives from:

```text
study_id
experiment_id
item_id
condition
layer
position
base_seed
```

through SHA-256. Record the full seed components in per-item logs.

## 18.2 One scoring contract

A single typed scoring spec must govern:

- G5 capability;
- teacher-forced endpoints;
- generation grading;
- counterfactual preference;
- audits;
- reproduction.

It must freeze:

- BOS/native units;
- prompt/answer concatenation;
- alias set and aggregation;
- whitespace policy;
- answer boundaries;
- generation normalization;
- max prompt/answer length;
- reasoning-delimiter parser version.

## 18.3 Per-item schema

Minimum fields:

```text
study_id
phase
tier
evidence_id
model_id
model_revision
lens_id
lens_sha256
config_sha256
bank_sha256
partition_sha256
item_id
fact_id
canonical_family
relation_group
variant
load
mode
phase_condition
condition
alias_set_hash
score_by_alias
score_aggregate
original_score
counterfactual_score
clean_answer_rank
clean_bridge_rank
selected_ids_hash
requested_k
selected_k
effective_rank
removed_energy
protected_rank
protected_overlap
lost_rank
seed_components
seed_value
stop_rule_flags
```

## 18.4 Do not store only summaries

Every figure and table must regenerate from immutable per-item rows. Store full intervention logs for a stratified sample and compressed hashes/aggregates for the remainder when size is large.

## 18.5 Input manifest and state refusal

Every producer must build an input manifest before loading a prior state. If any hash differs, exit and require a new output directory/evidence ID.

## 18.6 Environment lock

Phase 4 should add one of:

- a container digest plus constraints lock; or
- a fully pinned `uv.lock`/requirements lock with CUDA and model-library versions recorded.

The one-command bootstrap must verify versions rather than merely print them.

## 18.7 Path indirection

No scientific module should hardcode `/content/drive/...`. The current Phase 3 lens fitter does. Replace with URI/config resolution in Phase 4.

Required URI classes:

```text
repo://
drive://
model://
artifact://
```

## 18.8 Structured stop rules

Stop before outcomes when:

- baseline mismatch;
- model/tokenizer/lens hash mismatch;
- protected set misalignment;
- achieved rank outside tolerance;
- matched control dose mismatch;
- phase hook fires in the wrong phase;
- state manifest mismatch;
- sentinel drift;
- capability cohort unexpectedly changes;
- delivered perturbation fidelity fails;
- exact JVP disagrees with the smallest faithful secant.

Every stop writes an event and preserves state.

## 18.9 Review-level tests

Add tests for:

- stable seeds across processes;
- subprocesses with different `PYTHONHASHSEED` produce identical Phase 4 seeds;
- alias boundary matching;
- first-alias/logsumexp reproducibility;
- protected-answer stratum requires rank metadata;
- randomization plus-one p-values;
- exact sign-flip enumeration;
- proper bootstrap interval labeling;
- bridge true/distractor geometry matching;
- counterfactual target scoring;
- state/config hash refusal;
- phase parser edge cases;
- Gemma JVP/secant on a tiny nonlinear model with known curvature;
- OLMo-like linear positive control.


---

# 19. Concrete execution sequence

## Stage 0: snapshot the current state

Before changing code:

1. Record the current branch head.
2. Verify the Phase 3 freeze tag and partition hashes.
3. Hash the current Phase 3 report, handout, registry, primary/replication parquets, lenses, and base-lens artifact.
4. Export a live-evidence inventory.
5. Write `PHASE3_RELEASE_AUDIT_PLAN.md` with the five P0/P1 matters from Part I.
6. Tag a pre-audit snapshot if one does not already exist.

No Phase 3 outcome file is modified.

## Stage 1: Phase 3 protocol and inference audit

Implement and run:

```text
p3_protected_answer_audit.py
p3_control_seed_audit.py
p3_inference_audit.py
p3_bridge_geometry_audit.py
p3_alias_and_cohort_sensitivity.py
p3_n8_phase3_analysis.py
```

Required outputs:

```text
PHASE3_FREEZE_RECORD.md
PHASE3_RELEASE_AUDIT.md
PHASE3_STATE_OF_RECORD.md
phase3_release_manifest.json
jspace_phase3_final.pdf
```

Gate: every discrepancy is either corrected, bounded by sensitivity, or explicitly downgrades a claim.

## Stage 2: complete actual Phase 3 reproduction

Run N8-P3-L1. Then run stable-seed sentinels on all three models and one full Qwen Phase 3 cell.

If the original random-control realization cannot be reproduced exactly, say so. Exact baseline/J reproduction plus seed-ensemble control robustness is an acceptable and honest release result.

After this stage:

```text
git tag jspace-phase3-complete-v1
```

Create the Phase 4 branch and package.

## Stage 3: Phase 4 scaffold

Build:

```text
jspace_phase4/seeds.py
jspace_phase4/scoring4.py
jspace_phase4/manifests.py
jspace_phase4/state.py
jspace_phase4/stats4.py
jspace_phase4/phase_hooks.py
jspace_phase4/interventions.py
jspace_phase4/registry4.py
jspace_phase4/repro4.py
```

Port only validated Phase 3 components. Do not copy historical bugs for compatibility. Compatibility adapters belong in an import layer, not in the Phase 4 core.

Conformance gate before GPU:

- stable seed subprocess test;
- scoring golden across OLMo/Qwen/Gemma tokenizers;
- exact matched-control mechanical test;
- phase-hook sentinel;
- state mismatch refusal;
- randomization calibration;
- tiny-model JVP/secant test.

## Stage 4: finish the OLMo four-point trajectory

CPU/GPU sequence:

```text
1. download OLMo-3-32B-Think under stall guard
2. G5 on Phase 4 development banks
3. minimal span-safe grid + bridge mechanism
4. download base weights
5. G5 base
6. minimal span-safe grid + bridge mechanism using completed base lens
7. common-lens reruns on the same development slice
8. trajectory analysis and figure
9. decide whether intermediate SFT/DPO checkpoints are justified
```

Do not wait for Bank W authoring to start. CPU authoring and source verification can run in parallel.

## Stage 5: Qwen bridge mechanism completion

Build and gate Bank B. Run geometry-matched protection, bridge lesion, true rescue, counterfactual preference shift, and held-out replication.

This is the highest-value positive mechanism after the OLMo trajectory.

## Stage 6: Qwen mode factorial

Only after the parser and generation endpoints pass development gates. Run one complete model cell before adding more banks.

## Stage 7: Bank W

Author, shortcut-audit, capability-score, power-simulate, freeze, and run the OLMo pair plus Qwen. The load interaction decides whether the paper can use any working-memory language.

## Stage 8: fit-size and capacity symmetry

Run Qwen nested lenses first. Expand OLMo only if convergence remains open.

## Stage 9: receiver and broadcast localization

Use the strongest Qwen bridge families and one matched OLMo slice. Stop after one or two reproducible recipient paths rather than launching an unbounded component screen.

## Stage 10: Gemma bounded autopsy

Run exact JVP, localization, and at most the highest-priority mechanism interventions. Keep it methods-tier unless a new Gemma preregistration is frozen.

## Stage 11: independent reproduction and manuscript lock

A separate session with no report narrative:

- rebuilds the Phase 4 primary analysis;
- reruns model sentinels;
- reruns one full Qwen or OLMo primary cell;
- verifies the Gemma transport gate on a fixed prompt/direction subset;
- regenerates every final figure from raw records.

---

# 20. Suggested Phase 4 files and commands

## 20.1 Review and governance

```text
interpretability/jspaces/phases/phase4/reviews/jspace_lab_nextsteps_4_1.md
interpretability/jspaces/phases/phase4/reviews/PHASE4_PLAN_ACCEPTED.md
interpretability/jspaces/phases/phase4/preregistration/SCIENTIFIC_PREREGISTRATION_PHASE4_CANDIDATE.md
interpretability/jspaces/phases/phase4/preregistration/DEVIATIONS.md
interpretability/jspaces/phases/phase4/protocol/REPRO_CONTRACT_PHASE4.md
```

## 20.2 Experiments

```text
experiments/p4_olmo_lineage_grid.py
experiments/p4_lineage_dictionary.py
experiments/p4_lineage_analysis.py
experiments/p4_bridge_bank_gate.py
experiments/p4_bridge_geometry.py
experiments/p4_bridge_substitution.py
experiments/p4_qwen_mode_grid.py
experiments/p4_bank_w_author.py
experiments/p4_bank_w_gate.py
experiments/p4_bank_w_grid.py
experiments/p4_fit_size.py
experiments/p4_capacity_analysis.py
experiments/p4_receiver_search.py
experiments/p4_receiver_patch.py
experiments/p4_gemma_jvp_gate.py
experiments/p4_gemma_localize.py
experiments/p4_gemma_mechanism.py
experiments/p4_locked_analysis.py
experiments/p4_replication_analysis.py
experiments/p4_n8.py
```

## 20.3 Example stable command pattern

```bash
export JSPACE4_RUN_ROOT=/content/drive/MyDrive/interpret/special-lab-1/phase4_2026xxxx
export HF_HUB_CACHE=/content/hf_local

bash interpretability/jspaces/phases/phase4/repro.sh

python -m jspace_phase4.experiments.p4_olmo_lineage_grid \
  --config interpretability/jspaces/phases/phase4/configs/olmo3-base-dev.yaml
```

Every command must be recoverable from the evidence event. No ad-hoc notebook number enters the paper.

---

# 21. Implementation patterns

## 21.1 Exact protected-answer rank

```python
@torch.no_grad()
def clean_first_token_ranks(
    logits_at_boundary: torch.Tensor,
    alias_token_ids: list[int],
) -> dict[str, int]:
    row = logits_at_boundary.float()
    ranks = {
        str(tok_id): int((row > row[tok_id]).sum().item()) + 1
        for tok_id in alias_token_ids
    }
    return {
        "min_rank": min(ranks.values()),
        "ranks_json": ranks,
    }
```

Store the exact alias IDs and tokenizer hash. “Answer protected” must be a computable property, not a prose label.

## 21.2 Exact sign-flip enumeration

```python
import itertools
import numpy as np


def exact_signflip(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    obs = float(v.mean())
    stats = np.fromiter(
        (float((v * np.asarray(s)).mean())
         for s in itertools.product((-1.0, 1.0), repeat=len(v))),
        dtype=float,
        count=2 ** len(v),
    )
    p = float((np.abs(stats) >= abs(obs) - 1e-15).mean())
    return {"estimate": obs, "p": p, "n_patterns": int(len(stats))}
```

Use chunked/vectorized enumeration for larger `m`.

## 21.3 Monte Carlo p-value

```python
p = (int(np.count_nonzero(null_stats >= observed)) + 1) / (len(null_stats) + 1)
```

A printed zero is not a scientific p-value.

## 21.4 Counterfactual preference scoring

```python
def answer_preference(lp_original: float, lp_counterfactual: float) -> float:
    return lp_counterfactual - lp_original

substitution_effect = (
    answer_preference(cf_arm.orig_lp, cf_arm.cf_lp)
    - answer_preference(unrelated_arm.orig_lp, unrelated_arm.cf_lp)
)
```

Always report absolute calibration too. A preference flip where both answers collapse may be an intervention failure.

## 21.5 Bridge geometry matching

Pseudo-contract:

```python
@dataclass(frozen=True)
class AddedProtectionProfile:
    piece_count: int
    added_rank: int
    output_span_overlap: float
    selected_span_overlap: float
    bridge_answer_cosine: float
    activation_score_mean: float
    activation_score_max: float
```

Choose or construct a distractor whose profile falls within frozen tolerances. Log failure rather than silently accepting a poor match.

## 21.6 State header

```python
def validate_state_header(saved: dict, current: dict) -> None:
    mismatches = {
        k: (saved.get(k), current.get(k))
        for k in current
        if saved.get(k) != current.get(k)
    }
    if mismatches:
        raise RuntimeError(
            "Refusing to resume incompatible state: "
            + json.dumps(mismatches, sort_keys=True)
        )
```

## 21.7 Exact JVP gate sketch

```python
# Conceptual pattern. Adapt to the model wrapper and hook path.
def downstream_from_source(source_activation: torch.Tensor) -> torch.Tensor:
    return run_downstream_with_replaced_activation(source_activation)

clean = source.detach().float().requires_grad_(True)
direction = torch.nn.functional.normalize(direction.float(), dim=-1)

_, jvp = torch.autograd.functional.jvp(
    downstream_from_source,
    clean,
    direction,
    create_graph=False,
    strict=True,
)

for eps in eps_ladder:
    plus = downstream_from_source(clean + eps * direction)
    minus = downstream_from_source(clean - eps * direction)
    secant = (plus - minus) / (2 * eps)
    record(
        eps=eps,
        tangent_cos=cosine(secant, jvp),
        relative_error=(secant - jvp).norm() / secant.norm().clamp_min(1e-12),
    )
```

The actual harness must verify that the source replacement is the exact tensor consumed by the next sublayer and that the perturbation survives dtype conversion.

---

# 22. Compute budget and drop rules

Approximate 96 GB-class GPU budget:

| block | GPU estimate | priority |
|---|---:|---|
| Phase 3 release audit, control seeds + real N8-P3 | 4-8 h | mandatory |
| OLMo 3.0 Think + base G5/grids, base lens already done | 4-6 h | mandatory |
| Qwen bridge completion + replication | 3-5 h | mandatory |
| Qwen thinking-mode factorial | 3-5 h | high |
| Bank W, three models | 5-8 h | high |
| Qwen fit-size n=120/250/500/1000 | 6-12 h | medium-high |
| receiver localization/patching | 4-8 h | medium |
| Gemma exact gate + localization | 6-12 h | last |
| full program | roughly 35-60 h | |

## 22.1 Minimal paper-hardening set

A strong minimum is roughly 16-24 GPU hours:

1. Phase 3 release audit and actual N8-P3;
2. OLMo base and 3.0 Think trajectory points;
3. Qwen bridge target-preference test with replication;
4. Qwen mode factorial or Bank W, preferably both if budget permits;
5. Qwen n=120 versus published n=1000 lens sensitivity;
6. exact Gemma JVP gate on a compact positive-control set, only if time remains.

## 22.2 Drop order

Drop in this order:

```text
1. broad Gemma mechanism sweep
2. large receiver search
3. OLMo intermediate SFT/DPO checkpoints
4. OLMo 250/500 lens scaling after one representative convergence study
5. extra task families beyond a powered Bank W
6. secondary decoding temperatures
```

Never drop:

```text
Phase 3 protocol audit
stable seeds
bridge counterfactual target scoring
OLMo base + 3.0 Think trajectory
one controlled load test
actual Phase 3 reproduction
```

## 22.3 Complete cells, not confetti

Finish one model and one question end to end. Do not run 15% of eight workstreams and leave every conclusion with a new hole.

---

# 23. Phase 4 decision tree

## 23.1 OLMo lineage

- **No bridge mediation anywhere:** accessibility channel is lineage-stable; stop adding checkpoints.
- **Transition appears:** localize to adjacent training stage, then add only the necessary intermediate.
- **Own lens changes story, common lens does not:** representation-coordinate drift.
- **Common and own agree:** downstream usage changed.

## 23.2 Qwen bridge substitution

- **Counterfactual answer preference flips and replicates:** strong causal semantic bridge channel.
- **Original collapses but counterfactual does not rise:** disruption, not semantic substitution.
- **Geometry-matched distractor rescues equally:** P3-P3 was protection geometry, not bridge identity.
- **True bridge uniquely rescues but substitution fails:** bridge is necessary, but injected direction is not a valid writable code.

## 23.3 Thinking mode

- **Thinking-on reduces prefill dependence and shifts dependence to decode:** externalized/recomputed channel.
- **Thinking-on increases bridge dependence:** explicit reasoning consumes the internal bridge route.
- **Correct rationale uniquely rescues:** semantic externalization.
- **Filler rescues equally:** compute/time effect.

## 23.4 Bank W

- **Load interaction on Qwen:** upgrade from knowledge-access toward working-set channel, with scoped wording.
- **Load interaction only on OLMo:** OLMo may use J-space for in-context state but not parametric bridges.
- **No interaction with equivalence:** close the working-memory claim at tested loads/doses.
- **Prompt-length controls track effect:** sequence geometry confound.

## 23.5 Capacity

- **Qwen n=120 matches n=1000:** fit size ruled out.
- **Capacity changes but causal mechanism stable:** capacity metric is not mechanism.
- **Capacity and mechanism both move within a lineage/size series:** moderator hypothesis becomes testable.

## 23.6 Gemma

- **JVP matches tiny secants, moderate dose fails:** finite-radius curvature.
- **Prompt JVP works, mean fails:** context heterogeneity.
- **full-attention spikes:** routing mechanism.
- **MLP/norm linearization rescues:** local curvature mechanism.
- **late band passes with long lead:** relocate assay candidate.
- **late band passes only at emission boundary:** output staging.

---

# 24. Paper architecture after Phase 4

## 24.1 Recommended central paper

A strong paper should not be a chronological lab diary. Suggested structure:

### 1. Assay validity before model comparison

- output-token protection;
- label versus span protection;
- exact matched controls;
- scoring and clustering repairs;
- finite-dose transport gate;
- reproduction contract.

### 2. The leak-proof cross-model result

- Phase 2 and Phase 3 confirmatory/replication results;
- Qwen versus OLMo thick paired bank;
- prose/nonspecific caveat;
- capacity spread without causal overclaim.

### 3. What the channel carries

- Qwen bridge lesion/rescue/substitution;
- OLMo accessibility organization;
- OLMo lineage trajectory;
- Qwen mode factorial;
- Bank W load.

### 4. Where the method applies

- OLMo/Qwen transport validity;
- Gemma exact transport boundary;
- late-band/nonlinear alternatives.

### 5. General lessons

- verbalizable channels are model-dependent;
- output adjacency is not content specificity;
- capacity is not sufficient to infer workspace function;
- transport premise must be tested before a cross-model null is interpreted.

## 24.2 Claim ladder

### Already supportable after the Phase 3 audit, assuming the protected-stratum and seed audits remain robust

- A span-safe J-specific heavy tail exists on Qwen under exact dose controls and replicates across families.
- True bridge protection rescues Qwen composed-item performance relative to the frozen distractor arm.
- The thick OLMo pair is not distinguishable on the Phase 3 composition contrast.
- The tested Qwen effect is parametric knowledge-access shaped, not identified as a general working-memory workspace.
- Label-only output protection leaks substantially into the protected output span.
- No tested model shows clean task selectivity relative to the Phase 3 prose guard.

### Phase 4 target claims

- Qwen counterfactual bridge substitution moves answer preference toward the corresponding counterfactual answer.
- OLMo's channel organization changes, or remains stable, at a localized training stage.
- Thinking mode changes the phase in which the bridge channel is causally required.
- Controlled load either engages a J-specific working-set effect or is equivalent to no interaction within a frozen bound.
- Gemma's middle-band failure is localized to finite-dose curvature, context heterogeneity, or a specific architectural routing component.

### Prohibited without new evidence

- “Qwen has a global workspace.”
- “OLMo lacks working memory.”
- “Capacity causes the causal signature.”
- “Post-training monotonically creates a workspace.”
- “The counterfactual bridge swap made the model choose the wrong answer,” until the wrong answer is scored/generated.
- “Phase 3 was fully reproduced,” when the current N8 model cells are Phase 2 cells.
- “Gemma is non-differentiable.”
- “Gemma has no workspace.”
- “A nonlinear probe identifies a causal feature.”
- “The J-lens is faithful beyond the measured transport gate.”

---

# 25. Coding-agent queue

The coding agent should execute this queue in order and keep it as a checked ledger.

## Queue 1: Phase 3 closeout

- [ ] Create `PHASE3_FREEZE_RECORD.md`.
- [ ] Implement protected-answer rank audit.
- [ ] Recompute P3-P2 on exact stratum and replication side.
- [ ] Implement stable-seed helper and subprocess test.
- [ ] Run Phase 3 control-seed sensitivity.
- [ ] Exact-enumerate P3-P1 sign flips.
- [ ] Implement correctly named confidence intervals.
- [ ] Correct Monte Carlo p-value reporting.
- [ ] Audit bridge true/distractor geometry.
- [ ] Add original-versus-counterfactual scoring to mediation audit.
- [ ] Regrade cohort with boundary-safe matcher and report sensitivity.
- [ ] Correct replication-analysis tier metadata through an event.
- [ ] Build Phase 3 release manifest.
- [ ] Run N8-P3-L1/L2/L3.
- [ ] Write Phase 3 state-of-record report and final handout.
- [ ] Tag `jspace-phase3-complete-v1`.

## Queue 2: Phase 4 foundation

- [ ] Create Phase 4 branch/package/run root/registry.
- [ ] Port stable validated utilities only.
- [ ] Add config/state/input hash refusal.
- [ ] Add dependency lock and environment verification.
- [ ] Add all conformance tests in Section 18.9.
- [ ] Draft Phase 4 candidate preregistration with no frozen outcomes.

## Queue 3: OLMo lineage

- [ ] G5 OLMo-3-32B-Think.
- [ ] Development span-safe/bridge cell OLMo-3-32B-Think.
- [ ] G5 OLMo base.
- [ ] Development span-safe/bridge cell OLMo base using the completed lens.
- [ ] Common-lens reruns.
- [ ] Dictionary/projector trajectory.
- [ ] Accessibility, Bank F/S, prose, capacity, and bridge synthesis.
- [ ] Decide whether SFT/DPO intermediates are justified.

## Queue 4: Qwen mechanism

- [ ] Author and verify fresh Bank B.
- [ ] Geometry-match bridge distractors.
- [ ] Counterfactual preference and generation endpoints.
- [ ] Development dose gate with no outcome tuning.
- [ ] Power simulation.
- [ ] Freeze Bank B split and P4-P1.
- [ ] Confirmatory run.
- [ ] Replication run.
- [ ] Layer/position/phase localization.

## Queue 5: Qwen mode

- [ ] Official thinking template audit.
- [ ] Phase parser golden tests.
- [ ] Generation endpoint pilot.
- [ ] Correct/wrong/shuffled/filler rationale controls.
- [ ] Freeze P4-P2.
- [ ] Confirmatory and replication runs.

## Queue 6: Bank W

- [ ] Author six task superfamilies and nested loads.
- [ ] Shortcut audits and prompt-length controls.
- [ ] Capability gates on three models.
- [ ] Power simulation and SESOI.
- [ ] Family split and freeze P4-P3.
- [ ] Three-model grids.
- [ ] Replication.

## Queue 7: fit size and broadcast

- [ ] Qwen nested 120/250/500/1000 fits.
- [ ] Independent draws.
- [ ] Capacity and causal stability.
- [ ] Recipient screen on Qwen bridge items.
- [ ] Patch/lesion validation of top recipients.
- [ ] Matched OLMo contrast.

## Queue 8: Gemma

- [ ] Exact JVP/secant harness on tiny model.
- [ ] OLMo/Qwen positive controls.
- [ ] Gemma exact gate.
- [ ] Block and sublayer localization.
- [ ] Frozen routing/norm/gate interventions.
- [ ] Context clustering.
- [ ] Late-band gate.
- [ ] Stop or continue by Section 16.10.

## Queue 9: publication

- [ ] Regenerate all tables from raw rows.
- [ ] Independent analysis reproduction.
- [ ] Full cell reproduction.
- [ ] Artifact manifest.
- [ ] State-of-record reports only in main paper.
- [ ] Audit chronology in supplement.
- [ ] Release code and data instructions that work from a clean clone.

---

# 26. Phase 4 completion criteria

Phase 4 is complete when all of the following are true:

## Scientific

- Phase 3 release issues are resolved or transparently bounded.
- The OLMo four-point trajectory is complete under common and own lenses.
- Qwen bridge substitution is scored against the intended counterfactual answer and replicated.
- Qwen mode dependence is measured with generation endpoints.
- Bank W either finds a controlled load interaction or closes it with equivalence.
- Qwen fit-size asymmetry is resolved.
- At least one downstream bridge recipient is validated, or the search is honestly negative under declared bounds.
- Gemma receives the exact JVP gate and a bounded mechanism diagnosis.

## Statistical

- primary families are frozen before outcomes;
- every primary has an exact design-respecting test or a declared approximation;
- intervals are named correctly;
- small-cluster and leave-one-family-out sensitivities are reported;
- all Monte Carlo p-values use plus-one correction;
- null claims use equivalence.

## Engineering

- stable seed contract;
- no hardcoded machine paths;
- state/input hashes enforced;
- dependency lock verified;
- every result has immutable per-item records;
- every evidence event includes complete input hashes;
- clean-room analysis and model-cell reproduction pass.

## Writing

- main report contains no superseded numbers;
- freeze records are unambiguous;
- Phase 2, Phase 3, and Phase 4 claims are separately tiered;
- “workspace,” “knowledge-access channel,” “working-set channel,” and “output staging” are used according to explicit evidence gates;
- Gemma wording remains instrument-specific.

---

# 27. Final recommendation

Move to **Phase 4**.

Phase 3 has done enough to deserve closure: the leak-proof tail is confirmed and replicated, Qwen bridge rescue is confirmed, the thick task contrast is directionally informative but correctly unresolved, and the project has identified a more precise knowledge-access interpretation. Continuing to append lineage, mode, load, fit-size, and Gemma experiments under the Phase 3 label would blur the preregistered study boundary.

The first Phase 4 block should not load another 32B model immediately. Spend the opening block on the Phase 3 release audit, especially the protected-answer stratum, stable control seeds, exact P3-P1 inference, actual Phase 3 N8, and bridge counterfactual-target scoring. Then finish the OLMo base/3.0 trajectory, because the base lens is already paid for and that lineage is the most scientifically legible way to explain why OLMo and Qwen route composed knowledge differently.

After that, the sharpest scientific sequence is:

```text
OLMo lineage
-> Qwen counterfactual bridge replication
-> Qwen thinking-mode phase factorial
-> controlled Bank W load
-> fit-size/capacity symmetry
-> one receiver path
-> bounded Gemma exact-transport autopsy
```

That sequence turns the project from a large collection of careful ablations into a paper with a causal spine: **what the verbalizable channel carries, how models consume it, how training changes it, and when the linear instrument is allowed to speak.**
