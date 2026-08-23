# LogWarden Tier 1 preregistration

Status: authored before standard packet construction, target-model loading, or
target inference. This is a design commitment, not a result.

Governing inputs are `docs/SPEC.md`, `docs/SPEC_ADDENDUM.md`,
`config/campaigns/standard.json`, `config/agent-arms.json`,
`config/decode.json`, `config/baselines/rules-v1.json`, the standard scenario
catalog, the primary runbook corpus, and the pinned model registry. The campaign
freeze records their canonical hashes and the exact Git/database checkpoints.

## Question and estimand

On frozen, ground-truthed SQL Server incident packets, estimate the paired
episode-level change in acceptable-action accuracy and cost-weighted loss from
each local model/agent arm relative to the frozen deterministic B1 rules. Also
separate model quality from retrieval, tool, contract, orchestration, capture,
and systems-performance failures.

The observation unit is one incident episode. The inferential resampling unit
is `scenario_group_id`; repeated variants and events within a group are not
independent observations. Every compared arm sees the same eligible packet.

## Frozen population and splits

The intended standard corpus has 600 episodes: 60 `dev`, 60 `calibration`, 300
`test_id`, 120 `test_variant_holdout`, and 60 `test_unknown`. Scenario-template
groups are role-disjoint. `dev` is for plumbing and dependence estimation;
`calibration` alone fits calibration/abstention artifacts; test roles remain
unscored until the relevant configuration is frozen.

All ten families must have at least 30 episodes across `test_id` and
`test_variant_holdout`. Capture feasibility, exact packet count, exclusions,
and empirical power are freeze gates; counts will not be padded with fabricated
incidents.

## Models, arms, and baselines

Target profiles, in fixed order:

1. `muse-glimmer-30b`
2. `gemma-4-31b`
3. `olmo-3.1-32b-instruct`
4. `qwen-3.8-27b`

Only one chat model may be resident at a time. `qwen-smoke` is an explicitly
non-target plumbing/dev model and may run before target freeze.

Mandatory inference arms are `A-direct` and `A-tools` for every target and
quality-role packet. `A-rag` runs on the frozen retrieval-covered subset.
`A-router` is derived without a second model call: use B1 when B1 resolves the
packet, otherwise use that profile's `A-tools` result.

Baselines are B0 majority/no-action; B1 frozen deterministic rules; B2
retrieval-only under lexical, vector, and hybrid search; and evaluator-only B3
oracle packet classification. B3 cannot support a deployable-system claim.

The executable B0/B2/B3/router policy was added during standard SQL capture at
2026-08-23T08:45Z, before packet construction, target inference, target scoring,
or campaign freeze. B0 is mechanically fit on the predeclared dev role only;
B2 maps the top returned runbook's frozen class/severity metadata through a
frozen action map; B3 copies protected truth under `lw_lab` and is evaluator-only;
and the router selects resolved B1 rows, otherwise the matched A-tools row. This
timing cannot affect captured evidence, but is disclosed because executable
baseline policy was not present before capture began.

The executable Tier 1 control policy was likewise added during standard SQL
capture at 2026-08-23T09:00Z and before any packet or target output inspection.
It freezes the 96-episode subset transforms for error-number/signature masking
and shuffled runbooks, the first-48 sequential batching-invariance cell, and a
1,000-replicate family/role-stratified label-permutation null. The control file
is part of the campaign input manifest, so no transform or seed can change
after freeze.

## Transport and runtime

The primary decode is `primary-json-v3`: temperature 0, top-p 1, top-k 0,
min-p 0, repetition/presence/frequency penalties 1/0/0, seed 0, max tokens 900,
one completion, no stop strings, and non-streaming structured-JSON transport.
Each model's frozen chat-template kwargs remain profile-specific. The primary
request is a user-only operating-contract turn and does not send native tools
or a response-format constraint.

The agent budget is four model turns, four total tool calls, two calls to the
same tool, 4,000 characters per tool result, 12,000 tool-result characters in
total, and 180 seconds wall time. Raw request, raw response, reasoning channel,
tool result, validation, state transition, trace, service metric, GPU, SQL, and
queue evidence must be durable. Failed or missing runs are not imputed.

## Primary hypotheses

The Holm-corrected primary family contains the ten contrasts declared in the
specification:

1. each model/arm minus B1 on acceptable-action accuracy;
2. each model/arm minus B1 on cost-weighted loss;
3. `A-rag` minus `A-direct` on retrieval-covered episodes;
4. `A-tools` minus `A-rag` on tool/context-required episodes;
5. derived `A-router` minus `A-tools` on frozen quality/latency/token axes;
6. hybrid retrieval minus lexical and exact-vector retrieval;
7. packet correlation minus raw event-by-event correlation where measured;
8. calibrated minus raw confidence on Brier/ECE/selective risk;
9. replay versus bounded live-parity decision agreement;
10. exact versus ANN retrieval only if the governed ANN gate is later met.

Positive directional claims require a paired scenario-group bootstrap interval
for the difference that excludes zero after Holm correction, the corresponding
permutation control, complete evidence, and no applicable stop disposition.
A clean null is a successful experimental outcome.

## Metrics

Primary task metrics are end-to-end success, acceptable-action accuracy, and
cost-weighted loss. Supporting metrics include class accuracy/macro-F1,
severity exact and ordinal cost, preferred action, abstention/unknown recall,
contract/repair/failure rates, tool precision/recall and argument validity,
retrieval recall@1/3/5, MRR, nDCG and no-answer accuracy, citation/grounding,
latency phase decomposition, token/reasoning use, throughput/queue stability,
SQL cost, GPU utilization/power, and storage cost proxies.

Headline inference uses 10,000 paired grouped-bootstrap replicates and 1,000
within-group label permutations with fixed recorded seeds. The worst-case Holm
power check assumes base acceptable-action accuracy 0.80 and a 0.10 absolute
paired difference at alpha 0.05 and power 0.80. The final freeze requires the
simulation to be calibrated with dev-tier paired rows, not only design
assumptions.

## Calibration, missingness, and selection

Calibration models and abstention thresholds use only `calibration` episodes
and are hashed before test scoring. Report complete-case quality and end-to-end
quality with failures counted as failures. Report missingness by model, arm,
family, role, and failure stage. No failed inference is silently retried into a
different scientific cell; bounded transport retries remain part of the same
attempt and are retained.

The useful-triage gate is selective acceptable-action accuracy at least 0.85 at
coverage at least 0.50, with abstain/escalate recall at least 0.80 on
`test_unknown`, and no safety failure. It does not override paired/null tests.

## Retrieval and controls

Primary retrieval is exact SQL hybrid RRF over the frozen Qwen embedding
corpus. Lexical and exact-vector modes are mandatory comparators. Oracle and
shuffled retrieval are evaluator-only; shuffled results are plausible but
exclude acceptable runbooks. Past-incident memory is off.

The frozen 96-episode inference-control subset is deterministically
hash-stratified by family and regime from `test_id` plus `test_unknown`. Tier 1
controls are error-number masking, shuffled runbooks, retrieval ablation,
label permutation, leakage audit, and batching invariance. Label permutation
and the arm-ladder retrieval ablation use the full eligible test set.

## Safety and stopping

No live remediation, arbitrary SQL, write tool, or unapproved action execution
is permitted. A confirmed safety event yields `STOP_SAFETY`. Invalid capture,
matching, packet, ground-truth separation, leakage, or freeze evidence yields
`STOP_DATA`. A target model that fails residency/identity/decode/metrics/GPU
port gates yields `STOP_PORT`. Exhausted governed compute yields `STOP_BUDGET`
for remaining cells without rewriting completed outcomes.

## Declared adaptations and limitations before unblinding

- The standard catalog contains 600 rather than the older 384-episode target,
  following the addendum's power amendment. Impact: longer injection and model
  execution, better group/family support.
- SQL Server rejected `NO_EVENT_LOSS` for this event-file session. The capture
  uses capability-forced `ALLOW_SINGLE_EVENT_LOSS`; zero runtime dropped-event,
  dropped-buffer, blocked-fire, and target-failure counters are mandatory.
  Impact: completeness is proven empirically rather than promised by an
  unavailable retention mode.
- The resumable injector reconstructs its original schedule epoch from durable
  rows. This was implemented before the first standard episode. Impact: restart
  downtime does not reset or stretch planned offsets.
- B1 rules were authored after standard injection began, contrary to the
  addendum's literal pre-capture timing, but before standard packet construction,
  standard message inspection, test scoring, or target inference. They use
  runbooks and documented engine signatures. Impact: B1 remains outcome-blind,
  but reports must disclose the timing deviation and may not call it perfectly
  capture-time preregistered.
- The current packet projection omits duration/resource fields needed to
  distinguish successful resource-heavy RPCs from benign completions by rules
  alone. B1 therefore maps error-free completion evidence to no-action and its
  query-pressure misses measure packet/rule insufficiency, not an LLM win by
  construction.
- Mac Foundry/MLX results are a separate portability plane with a different
  runtime and small episode set. They are not pooled with the frozen Colab
  four-model campaign.

## Blinding statement

At this document's authorship, operational progress counts, health counters,
development gates, runbook content, scenario definitions, and Mac-plane results
had been inspected. No standard incident packet, captured standard message,
target-model standard output, or target test score had been inspected. The
freeze command will hash this document and record any later amendment
append-only; it will not overwrite this statement.
