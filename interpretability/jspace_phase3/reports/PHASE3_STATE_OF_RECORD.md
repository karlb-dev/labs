# J-space Phase 3 — state of record

Status: release candidate, 2026-07-31.  
Frozen design tag: `jspace-phase3-freeze-v1` (`df4d45a`).  
Release-audit entry tag: `jspace-phase3-pre-release-audit-v1`
(`660047d`).  
Planned completion tag: `jspace-phase3-complete-v1`.

This document supersedes the chronological interpretation in
`REPORT_PHASE3.md`. It does not rewrite any frozen outcome. Corrections and
sensitivities are new event-sourced evidence in
`reports/evidence_events.jsonl`.

## 1. Frozen design

Phase 3 uses a thick paired bank: every fact has direct and composed prompts
with the same answer and frozen accepted-alias set. The confirmatory and
replication partitions are disjoint by canonical family. The primary
intervention is span-safe J-space ablation; its comparator consumes the same
per-position rank and energy profile in an instant matched random subspace.

The three primary models and frozen revisions are:

| model | revision | J layers |
|---|---|---|
| OLMo 3.1 32B Think | `832c3f543499af8fe68b88359501de9cb7840544` | 20, 22, …, 44 |
| OLMo 3.1 32B Instruct | `ac0587e4a7744a551c059d8cd17ba220bc940dae` | 20, 22, …, 44 |
| Qwen3.6-27B | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | 20, 22, …, 44 |

The frozen confirmatory cells contain 170 Think, 164 Instruct, and 188 Qwen
direct/composed items. The held-out replication cells contain 152, 148, and
190 items. The P3-P1 unit is a paired fact, then an equal-weight canonical
family. P3-P2 is the Qwen excess rate of a span-safe J effect below −1 nat
relative to its matched control. P3-P3 is the Qwen true-minus-distractor
bridge-protection rescue.

## 2. Final primary and replication results

The historical matched controls used Python `hash(item_id)` without a recorded
`PYTHONHASHSEED`. Baseline and J arms are deterministic; only the control
realization is unrecoverable. The state-of-record reconstruction replaces
Qwen’s control with the explicit `sha256-v1` realization at seed 31337. The
OLMo control rows remain the immutable historical realizations. This matters
for P3-P1 and is why it is descriptive.

| result | confirmatory state of record | held-out replication | status |
|---|---:|---:|---|
| P3-P1, Qwen − mean OLMo composition specificity | −0.271183, exact p=0.057892, 17 families | −0.197020, exact p=0.219482, 17 families | descriptive, negative, seed-sensitive |
| P3-P2, Qwen span-safe tail excess at −1 nat | +0.095833, plus-one p=1/100001, 188 items / 26 families | +0.102083, plus-one p=1/100001, 190 items / 28 families | confirmatory and replicated |
| P3-P3, Qwen true-vs-distractor bridge rescue | +0.431367 nats, plus-one p=0.009180, 94 items / 26 families | not collected | confirmatory on the frozen chosen-distractor contrast; not replicated |

At the historical Qwen control realization, P3-P1 was −0.260954 with exact
p=0.062332. Its randomization-compatible 95% confidence set is
[−0.535135, +0.014565], and the correctly named wild-cluster percentile-t
interval is [−0.537109, +0.015201]. The normal `estimate ± 1.96 SE` interval
[−0.515271, −0.006637] is retained only as a small-cluster approximation and
is not called a bootstrap interval. Evidence:
`p3-inference-audit-v1`,
`p3-control-seed-contract-audit-v2`,
`p3-n8-p3-level3-qwen36-27b-v1`.

### Protected-answer conformance

The P3-P2 result is unchanged at the preregistered protected-answer stratum:

- all confirmatory Qwen items: +0.095833, 188 items;
- exact scored-alias clean rank ≤10: +0.095833, 183 items;
- minimum rank over any accepted alias ≤10: +0.095833, 185 items.

All three have plus-one p=1/100001. On replication, the protected views are
+0.098895 on 188 items, versus +0.102083 on all 190. Evidence:
`p3-protocol-audit-protected-answer-qwen-v1`.

## 3. Release-audit corrections

1. **Control seeds.** Five full-cohort Qwen realizations (11, 101, 1009,
   4242, 31337) leave P3-P2 unchanged but move P3-P1 across the 0.05 decision
   threshold. P3-P1 therefore receives no inferential “near-miss” wording.
2. **Inference labels.** P3-P1 now has exact 2^17 sign-flip enumeration,
   randomization inversion, and separately named percentile-t and normal
   intervals. P3-P2/P3-P3 Monte Carlo p-values use the plus-one rule.
3. **Reproduction scope.** Historical N8 jobs are labeled N8-P2. The actual
   Phase 3 ladder is complete: N8-P3-L1 reproduced 61/61 quantities exactly;
   stratified L2 cells reproduced deterministic arms for all three models;
   L3 reproduced all 188 Qwen rows, including exact deterministic arms and
   exact stable control.
4. **Replication metadata.** `p3-replication-analysis-v1` is corrected
   event-wise from `phase3-confirmatory` to `phase3-replication`.
5. **Generation boundaries.** Word-boundary grading rejects “India” in
   “Indian” and “Dutch” in “Dutchman”; Unicode NFKD folding recovers Bogotá
   and İstanbul variants. A stratified 100-positive/100-negative hand audit
   has zero disagreements.
6. **Registry immutability.** The obsolete historical Qwen N8-P2-L2 event is
   withdrawn because its producer reused an output path later overwritten by
   L3. The distinct N8-P3-L2-v2 and N8-P3-L3-v1 artifacts are intact.

## 4. Sensitivity results

### Boundary and cohort selection

Boundary-safe grading changes four unsafe substring positives and recovers
four Unicode-accent matches among 2,916 G5 rows. At Qwen stable-control seed
31337, the confirmatory P3-P1 estimate changes from −0.271183 to −0.270160.
Boundary-safe P3-P2 is +0.095833 on 186 items (p=2/100001), and P3-P3 is
+0.428230 on 93 items (p=0.009420). Direct-capable, preference-margin, and
all-source-verified views are reported only where frozen outcomes identify
them; missing eligible outcomes are never treated as observed. Evidence:
`p3-boundary-cohort-sensitivity-v2`.

### Accepted aliases

A pre-outcome sensitivity subset contains 20 common confirmatory facts, all
17 shared families, and all four multi-alias facts. Each model scored 52 alias
cells on CUDA. First-alias deterministic replay errors are below 0.000214
nats, and all tokenizer-level accepted sets are prefix-disjoint.

On this deliberately enriched subset:

| endpoint | P3-P1 estimate | change from stable first |
|---|---:|---:|
| historical first alias | −0.361097 | not comparable: historical control |
| stable first alias | −0.267301 | reference |
| canonical alias | −0.383906 | −0.116605 |
| prefix-disjoint logsumexp | −0.246098 | +0.021203 |
| max alias, diagnostic | −0.267357 | −0.000056 |

The canonical change is concentrated in the two facts whose canonical alias
differs from the first alias. The Qwen P3-P2 subset tail statistic is
identical under all five views. Phase 4 will use frozen prefix-disjoint
logsumexp prospectively. Evidence:
`p3-alias-endpoint-cross-model-v1` and its three model cells.

## 5. Bridge mechanism closeout

### Confirmatory chosen-distractor rescue

The corrected geometry audit (`p3-bridge-geometry-qwen36-27b-v2`) replays all
four frozen arms exactly over 94 facts and 67,834 layer-position rows. It has
zero selected/protected-rank accounting failures, zero final
selected/protected overlap, and complete survival of protected bridge
directions.

The raw rescue is +0.431367
[+0.132018, +0.763437] by family bootstrap. Nested leave-one-family-out
geometry prediction has cross-fit R²=−0.0947. Removing that prediction leaves
+0.403816 [+0.105071, +0.734497], p=0.01854. Thus the measured rank, energy,
piece-count, overlap, and answer-cosine geometry does not explain away the
rescue. The exact all-site geometry-matched subset has only 10 facts /
5 families (+0.450282, exact p=0.1875) and is underpowered.

This remains a true bridge versus chosen distractor result. The distractor
was not randomized, and no untouched-family P3-P3 replication exists.

### Counterfactual semantic endpoint

On the existing 40-fact / 13-family mediation cohort,
counterfactual-bridge injection shifts
`LP(counterfactual answer) − LP(original answer)` by +8.582031 nats
[+5.077661, +12.122991], exact p=0.000488. The shift is +4.765142 nats
relative to a geometry-selected unrelated injection. Greedy generation moves
from 32 original / 0 counterfactual hits at baseline to 7 original /
15 counterfactual hits under the swap.

However, direct counterfactual-answer-direction injection itself shifts
preference by +7.240 nats. Counterfactual bridge minus counterfactual answer
direction is only +1.342254 [−1.593275, +4.482051], p=0.419. The endpoint
proves intended semantic movement, but does not isolate an abstract bridge
channel from a downstream answer-direction route. It is development evidence,
not untouched-family replication. Evidence:
`p3-bridge-swap-endpoint-qwen36-27b-v1`.

## 6. Exact claim ladder

1. **Confirmatory and replicated:** on Qwen, span-safe J ablation produces a
   content-specific heavy tail beyond an exact matched control. The result
   survives clean answer protection, control seeds, boundary-safe cohort
   grading, and accepted-alias scoring.
2. **Confirmatory, not replicated:** protecting the true bridge rescues Qwen
   more than protecting the frozen chosen distractor. Measured geometry does
   not account for the contrast.
3. **Development:** counterfactual bridge injection moves answer probability
   and generation toward the intended counterfactual. It is not yet separable
   from direct answer-direction injection.
4. **Descriptive:** Qwen’s direct-to-composed specificity change is more
   negative than the OLMo pair’s, but P3-P1 is underpowered and
   control-seed-sensitive.
5. **Negative boundary:** no primary model earns a selective-global-workspace
   claim. Prose damage exceeds task damage in standardized units, and Qwen’s
   Bank-S composition term is null. “Knowledge-access channel” is the
   strongest licensed noun.

## 7. Limitations

- P3-P1 has only 17 intersection families and an unrecoverable historical
  matched-control salt for each frozen model.
- P3-P3 has no held-out-family replication and compares a true bridge with a
  semantically chosen, not randomized, distractor.
- The semantic swap is post-freeze development work on an existing cohort.
- Alias sensitivity contains only four shared multi-alias facts; canonical
  surface sensitivity is consequently localized rather than broadly
  estimated.
- Capability filtering conditions on pre-intervention model behavior.
  Superset estimands with missing intervention outcomes are not identified.
- The intervention is not behaviorally selective on the prose guard, so
  “global workspace” and broad working-memory language remain unlicensed.

## 8. Artifact map

| purpose | live evidence | durable location under `phase3_20260729/` |
|---|---|---|
| frozen primary/replication inference | `p3-inference-audit-v1` | `metrics/cross_model/release_audit/` |
| protected-answer ranks | `p3-protocol-audit-protected-answer-qwen-v1` | `metrics/qwen36-27b/release_audit/protected_answer/` |
| five Qwen control seeds | `p3-control-seed-contract-audit-v2` | `metrics/qwen36-27b/release_audit/control_seed/` |
| actual Phase 3 N8 ladder | `p3-n8-p3-level1-v1`; three L2 cells; Qwen L3 | `n8_phase3/` and `metrics/cross_model/release_audit/n8_phase3/` |
| boundary/cohort sensitivity | `p3-boundary-cohort-sensitivity-v2` | `metrics/cross_model/release_audit/alias_cohort_sensitivity_v2/` |
| accepted-alias sensitivity | `p3-alias-endpoint-cross-model-v1` | `metrics/cross_model/release_audit/alias_endpoint/` |
| bridge geometry | `p3-bridge-geometry-qwen36-27b-v2` | `metrics/qwen36-27b/release_audit/bridge_geometry_v2/` |
| semantic bridge swap | `p3-bridge-swap-endpoint-qwen36-27b-v1` | `metrics/qwen36-27b/release_audit/bridge_swap_endpoint/` |
| release manifest | `p3-release-manifest-v1` | `manifests/phase3_release_manifest.json` |
| final handout | `p3-state-of-record-release-v1` | repo `reports/handout/jspace_phase3_final.pdf` and run root |

The release manifest hash-pins the full live-evidence inventory, model and
tokenizer identities, banks, partition, lenses, raw parquets, environment,
seed contract, figures, and this report.
