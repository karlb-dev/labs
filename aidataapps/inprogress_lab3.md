# Lab 03 in progress — LogWarden

Last manually updated: 2026-08-23 16:28 UTC

Read `resume.md` first for worktree, recovery, and evidence rules. This file is
the volatile state of Lab 3 and must be refreshed before and after long jobs and
at every scientific evidence boundary.

## Ownership and objective

- owner: the Lab 3 Codex session on the clean Colab VM started 2026-08-23 UTC
- worktree: `/content/worktrees/aidataapps-logwarden`
- branch: `aidataapps-logwarden`
- remote branch: `origin/aidataapps-logwarden`
- base branch: `origin/aidataapps-modelprint`
- base commit: `88ea443092cd27226272776e6c6fcf8fbce329de`
- lab directory: `/content/worktrees/aidataapps-logwarden/aidataapps/logwarden`
- Drive root: `/content/drive/MyDrive/aidataapps/lab03`
- governed source plan:
  `/content/drive/MyDrive/aidataapps/lab03/aidataapps_logwarden_lab_3_spec.md`
- governing addendum:
  `/content/drive/MyDrive/aidataapps/lab03/aidataapps_logwarden_lab_3_spec_addendum.md`

The objective is the addendum's Tier 1 Minimum State of Record: construct and
freeze a ground-truthed SQL Server incident corpus, run the governed four-model
replay benchmark and baselines, score it with paired statistics and controls,
and close with a reproducible row-only and database-restore state of record.
The addendum wins over the base specification. Narrow implementation/runtime
adjustments are allowed when required to reach completion, but every adjustment
and its effect on the evidence ceiling must be recorded append-only in
`EXPERIMENT_LOG.md`.

## Current resumption state (authoritative)

- Active run: `logwarden-smoke-20260823T031714Z`.
- The standard campaign is frozen and target inference is authorized. Freeze
  hash: `e105cfdd5af5345464853406c8d707018232f3326c909ca476970fb7137cbcf6`;
  receipt:
  `1819a96b69cb90c73c450aa2da5376bf6ab565c3d20f9293192746e5fc4e60df`;
  packet set:
  `98796f59549b3e529fef5c7363b7294c917ea098b4d4a24742626b5c5d179835`;
  frozen control subset:
  `225e6f1d88aefe64cf7d07139155cab8d32e337770004acd22f361f23986ed8a`.
- Qwen smoke, power, final pre-freeze telemetry reconciliation, retrieval
  evaluation, search freeze, database restore test, run archive, and Drive
  mirror all passed. The global pre-freeze reconciliation receipt is
  `2a5f0a4e39bb2831eec506067de3186ab2eb1aeb2965e322161b69f5fb0287c8`.
- `muse-glimmer-30b` is the active governed residency. Exact container
  `9602d24544f974bdf1d0287136d7bb9c8a3da1dec67aa33dabf25d10bb3e9a40`
  is running on port 8010 with the frozen image, model, revision, profile hash,
  16K context, max-num-seqs 64, and batch-invariant kernels. The first cold
  cache startup stopped before HTTP/model requests in the same tokenizer race
  retained by Lab 2; the settled-cache service passed two formal nine-call
  gates with zero length/error/preemption outcomes and exact repeated output.
  PASS receipts:
  `4fd5df4af5c5fe6a3d73cfa759bb2de372e59e027687ad144b88f15952a6ad56`
  and
  `4fa35f3fd6dba3cf83d3cdf07b087e753e72e2040beedfe6ecce623af2ac16b3`.
  The cold failure and one log-classifier false positive remain retained and
  are disclosed in `EXPERIMENT_LOG.md`; neither produced scientific rows.
- Muse whole-residency sampling is closed. Epoch
  `26eda471-0ceb-476f-be48-6304e7ed6c4b` contributed 6,276/6,276 records; the
  global reconciliation then passed across 193 journals, 119,595 records,
  8,640 raw artifacts, 6,001 closed traces, 31,220 closed spans, 4,243 paired
  model requests/responses, 2,725,869 metric samples, and 11,285 raw metric
  snapshots. Receipt:
  `fd256614a7fe153fbbe17b88acb8a72589ccd5db63050b7d644e38de8fadad2e`.
  Qwen embedding remains on port 8011, SQL Server on port 1434, and the
  rootless Docker daemon is supervised in retained exec cell `64558`.
- Muse calibration primary replay passed all integrity gates: 180 terminal
  cells, 166 decisions, 14 retained end-to-end failures, 319/319 successful
  stop-finished HTTP requests, zero service error/length/preemption outcomes,
  139 tool calls, zero retained leases, and zero open/unlinked spans. Receipt:
  `8a6cb7cba39bec1cc934b2aa2721ec1d783f636b3577b62c0fe057f07c51f982`.
  The 40-minute inference window reflects three effective workers, not 16,
  because 13 workers retired on transient empty `READPAST` claims. This affects
  calibration throughput comparability only; inputs, outputs, and quality rows
  are intact. The defect and impact are append-only in `EXPERIMENT_LOG.md`.
- Before opening test predictions, source was hardened with selected-queue
  availability probes, bounded/instrumented deterministic claim backoff, and
  replay-sized SQL pools. The real gate reproduced three transient empties and
  recovered all of them, yielding 16 distinct claims for 16 workers in 189.12
  ms; all 8 SQL integration cases and 38 files / 134 unit tests passed. SQL
  receipt:
  `0cb7233598d78ccb665d0df8cc06934741ee48eab70e82099b55da2b335eba73`;
  durable commit `1c471f6`.
- Source and evidence are being checkpointed after protected Muse scoring;
  calibration evidence is durable through `d0bccd5`, queue hardening through
  `1c471f6`, and the watchdog has independently retained the protected replay.
  Foundry/Mac/report support is merged; Linux Foundry validation
  remains deferred until the four governed GPU profiles finish.
- Muse calibration chronology is closed before test access: `A-router` derived
  60 rows (receipt `b2b0bdea164ea862da2da7321f08e5692dbbb51a0acaf4680259a68118699904`),
  all 240 four-arm rows scored (receipt
  `a9555119068794d857dcefaaa6bf6cbd21f472ecbb87b3f3ee41def3351f3741`),
  and four calibration-only models hash-locked with test authorization
  (receipt `61b5ccffadacffc05e6a97f0aced1a0db3a7acae25013e6f3c88c4252e5dcb16`).
  The guard confirmed zero Muse test predictions at fit time.
- Muse protected primary replay passed: all 1,370 governed cells completed,
  with 1,249 decisions, 121 retained failures, and 2,563 successful model
  requests. All 16 workers claimed 82–90 cells; four transient empty claims
  each recovered after one 20–32 ms retry. vLLM recorded 2,562 stop finishes,
  one retained length finish, zero errors/preemptions, 4,675,075 prompt tokens,
  and 674,049 generation tokens. Receipt:
  `77c40337aac5ec4379845bcef8076bc5e43652790e65c0cf09daded52f6eaa9d`.
  Deterministic B1 was then completed for 480 protected rows after repairing a
  post-freeze lifecycle guard (ordering/resumability only; no scientific input
  or model output changed). A-router derived all 480 rows and scoring persisted
  all 1,850 four-arm predictions plus 11,325 tool and 1,850 retrieval scores.
  Score receipt:
  `538311178007a1b5b32e7c2a0ef9c357dd760ea4120d66a1c0923d4cc0eb6c02`.
  The 96-cell error-number-mask and shuffled-runbook controls have both passed
  replay, scoring, and paired comparison with zero model-service errors or
  preemptions. Masking produced 50.0% semantic decision agreement and a
  -0.041667 mean action-score delta; shuffled retrieval produced 51.0417%
  agreement and a -0.1875 delta. Comparison receipts:
  `a566c9b7046a69446a1c71f3c4b01d65386845f2360781137d2e20d2bd27fe42`
  and `750f09ae8244eb9aceee4f9245c9b090cdbe9a01e5aa3a6c850844d0c529f7bb`.
  The 48-cell single-worker batching replay also passed (44 decisions, four
  retained agent failures, 129/129 successful stop-finished requests, zero
  service errors/preemptions), then scored and compared. Its literal full-agent
  result was `NON_INVARIANT`, but a formal audit proved that later prompts embed
  new agent/tool/retrieval execution IDs and raw vLLM bodies embed response
  IDs/timestamps. All 48/48 byte-identical first-turn requests and all 48 exact
  request pairs produced identical normalized choices/reasoning. The governing
  interpretation is therefore `AGENT_INPUT_DRIFT_REQUEST_LEVEL_INVARIANT`:
  observed request-level invariance, full-agent invariance not identifiable.
  Original rows remain immutable. Diagnostic receipt:
  `7810da399de51d80e99b133a8490feff11ab7ed3b24821429786feaac43c298d`.
  Muse's verified profile checkpoint is pinned at receipt
  `18447e0da29362059b975ae2cbec930bc77b4d3aad864105311de6543ec8e8de`.
  Bounded retention passed and removed 28,350,983,266 bytes of redundant
  local/Drive/SQL-staging copies while preserving the frozen boundary, Muse
  boundary, two newest rolling checkpoints, and two recovery bundles. Receipt:
  `ba8843aa7f452b553d14772d334372da2c38641940a158749d3f586bea76eb24`.
  Muse is stopped and its reproducible cache plus obsolete qwen-smoke cache
  were evicted; all evidence remains retained. Gemma's exact 58.25 GiB pinned
  checkpoint downloaded and loaded, but its first formal cold port gate stopped
  before HTTP/model requests: a 16K request needs 13.76 GiB KV while the frozen
  0.78 resource cap exposed 13.22 GiB. STOP_PORT receipt:
  `103e7dabd7506d125a60dcd8e5fa4e323072d97539669105c7d425d08c9b6133`.
  A runtime-only 0.79 GPU-memory override is implemented and tested without
  changing the frozen model/profile hash, context, prompts, decode, or weights;
  its +1-point resource difference must be disclosed in performance results.
  The settled-cache service then passed two formal nine-call gates: 18/18
  successful requests and stop finishes, zero length/error/preemption outcomes,
  and 82,766 MiB observed total GPU memory. PASS receipts:
  `0a9e9adb19f74d70d616c981061ef4acbda51e2820009b5a4e925f5c41f64f01`
  and
  `ed37c268649a713231c585b6a6710fb2a35031cf876fdcdfc557d8d5afbf5b4a`.
  Repeated exact-output rates were 1/6 and 2/6 and remain a measured Gemma
  property for the formal batching control. Gemma calibration primary replay
  then passed all 180 cells with 180 decisions, zero failures, 250/250
  successful stop-finished requests, zero length/error/preemption outcomes,
  384,158 prompt tokens, and 35,935 generation tokens. Receipt:
  `e6a9cbc9cc0ed523b9751c6d6db25a8342db8d29bb5f2657ce8f69e310e844d7`.
  A-router derived 60 rows (receipt
  `a9891d4e5224dce842a700fba7df56448204e7024fd4ff4fcecfc1ae63a6244a`),
  all 240 calibration predictions were scored (receipt
  `8ad054196fb90c9c8e26e0cc27fb57f53f608819cde6832d7bbe2fbf33d9ee32`),
  and four calibration-only models were fit with zero prior Gemma test
  predictions (receipt
  `cfaae197a0c42323f5f73ea405b6fd3e2dcf4a3e70d0d1d4ae489f0451dbbe83`).
  Next: launch Gemma protected primary replay (1,370 cells, 16 workers), then
  derive router and score. Do not relaunch completed Muse/Gemma calibration or
  the retained failed Gemma cold gate.

```bash
cd /content/worktrees/aidataapps-logwarden/aidataapps/logwarden
source scripts/runtime-env.sh
npm run telemetry:reconcile
```

## Historical setup context (superseded where conflicting)

- Phase: LW-0/LW-1 complete; LW-2/LW-3 smoke capture, packet, recovery,
  agent-observability, least-privilege tools, real eight-family injectors,
  exact-correlation/context-snapshot gates, the real Qwen embedding vLLM port
  gate, all 480 corpus embeddings, the three-mode hybrid retrieval development
  gate, the fully persisted bounded agent loop, executable baselines, and
  inference-control instrumentation passed. Standard capture is active;
  held-out retrieval assessment, corpus freeze, and chat-model canary remain.
- The dedicated worktree was created from the exact current Lab 2 remote head.
- The new branch was pushed to GitHub and tracks its own remote branch.
- The repository was clean at branch creation.
- The Lab 3 source tree, control/workload databases, initial smoke run, custom
  XE capture session, Query Store configuration, and two-principal security
  model exist and have passed their foundation gates.
- The pinned Qwen 0.6B embedding weights are downloaded and vLLM is resident
  on GPU at port 8011. It generated and verified the complete primary corpus;
  no chat-model weights are loaded.
- The complete spec/addendum and predecessor/reference inputs were read and
  hashed before implementation.
- A user-directed pre-inference observability gate now promotes detailed
  agent/vLLM/queue/SQL/XE/GPU telemetry and dual file/SQL persistence before
  any long model campaign; see `logwarden/docs/OBSERVABILITY_CONTRACT.md`.
- Thirty hash-locked control migrations are applied. Migration 031 for
  inference/retrieval provenance and migration 032 for Tier 1 control
  provenance are authored but intentionally held until standard capture
  completes; seven versioned
  server/XE/security assets now apply idempotently. `npm run doctor` passes all
  required probes; `npm run test:sql` passes 8/8 integration cases;
  `npm run check` passes 30 test files and 98 unit tests, including resumable
  schedule timing, pinned chat-service construction, chat Prometheus deltas,
  and separate reasoning-channel retention. The capture-specific
  XE predicate excludes agent/ingest traffic.
- Development schedule `smoke-v1` injected ten safe scenarios; all ten cleanup
  gates and all required XE/ERRORLOG evidence rules passed. Capture verification
  receipt: `7271b30792a9e054b8f95cedb057df153dda4434517f348b8894cc11fe6bbd5`.
- Raw ERRORLOG replay is idempotent (0 inserted / 461 duplicates on immediate
  replay). XE uses file + block offset + within-block ordinal after runtime
  discovery that offsets are block-scoped.
- Ten development packets have separately protected evaluator truth and passed
  the leakage audit with zero findings. Packet build/audit receipts are
  `06670cc91548962090843e921f0b72d0f58e38544f79b2fd0ffb9768fb749e11`
  and `f6cc4310ac038a320106c94ad3d43912e9737fd840043c85b63ccc7a698fbd74`.
- A full no-model systems sample persisted GPU, SQL, queue, XE, host, and raw
  endpoint-unavailable observations. Journal-to-SQL replay is hash validated
  and idempotent; latest receipts are
  `f19f2884678a01d417f41dbe559f5be4416adc2a46012b0cde6cb3c4f4e89867`
  (sample) and
  `430d53f721b081915e88d5d79ece7362673e7e611e9332b6e7153e9b342d3ab4`
  (journal ingest).
- The pre-inference agent observability gates now pass: nine fake-gateway
  terminal routes with exact raw-byte/SQL hashes
  (`fdce92eeeb8344b52c02455a6d244ee9d9839bc57c2f556f01be5caae0f5d474`),
  committed-batch crash/replay
  (`ac60e25a385f2ecfedc1bec6d4fd33dc92de9cbb78f6d0c5935bcda529d20660`),
  two-process sampler restart
  (`c30e80918b27592426a3657cb0e6c64fe97aa4a09057eda935deff97ba290218`),
  and p95 added synthetic instrumentation latency of 1.497 ms
  (`3888e28dbd673f71e75ef09bf34451e9b15c4d3196e8c0db2a4c66a2baa6efea`).
  Global journal/SQL/raw/trace reconciliation covers 252 events and 66 raw
  artifacts with receipt
  `8ffc7bd73a43d50d64113cabfe8fc9f0e70db105b0271acd308230cc6c89c505`;
  the derived file rebuilt byte-for-byte.
- ERRORLOG `sp_cycle_errorlog` recovery passed with a stable pre-roll key,
  one unique post-roll marker, and 599/599 duplicates on immediate replay;
  receipt `594b46485c6865068c2093c311190b09eb900406c51b76d2f972704a73087d1e`.
- SQL/container restart recovery passed with distinct container/SQL start
  identities, stable pre-restart source identity, a unique post-restart row,
  and 997/997 duplicates on immediate replay; receipt
  `f6688d4472af0db6947076097a27be7d14cd58490b736d9dc442fb2c2adc2c99`.
- The seven-tool `tools-v1` registry and its SQL enforcement layer passed nine
  positive and eleven negative least-privilege cases. Direct evaluator, snapshot,
  runbook-table, queue, ingestion, DDL, server-DMV, and `msdb` backup-table
  bypasses are denied. Receipt:
  `f5965e712ba2edfe542ca767f042ecc29d81d985af3b85f8bdb6cb2e80de48cf`;
  registry hash:
  `25c79c34cd382bba6bb1f9139401aac8bfea2f603bd83d137b0eb209ddfac6bb`.
  The registry remains marked building/unfrozen until standard runbooks close.
- The deterministic `primary-v1` knowledge corpus now contains 60 original
  MIT-licensed guides and 480 heading-aware chunks across all ten classes, with
  exact source/body/chunk hashes and zero lab-identifier leakage findings.
  Full-text indexes both headings and content; all 11 restricted-agent lexical
  canaries retrieved the intended family. Manifest/build/gate hashes are
  `b734ba4c05ff3e548b779b1f9718e8956e4f0f9bb45a8246492664f88a0e66df`,
  `d8b6b81aff2811460fdee6a5469ceb69540381720ceaf9f961af7c149e2f2d79`,
  and `7e8c38b4218dae4b43c27e95cd522ae669f026fbe94249c3890e7d8cc51808be`.
  The corpus remains unfrozen pending the standard scenario leakage audit and
  held-out retrieval assessment; its Qwen embeddings and hybrid mechanics are
  complete.
- The live Qwen embedding port gate passed with receipt
  `d9f7b756c907a4aa9516a1c47875dec2a5e7d932b4bc4a721c93bb3e93492323`.
  It proved exact image/model/revision/runner identity, health/model listing,
  four HTTP calls and six ordered finite normalized 1024-dimensional outputs,
  exact warmed-repeat identity, raw request/response durability, 32
  journal-to-SQL records, and before/midpoint/after vLLM plus GPU snapshots.
  Counter deltas were 4 HTTP, 6 success, 108 prompt tokens, 6 latency, 0 error,
  and 0 preemption; midpoint GPU evidence showed the EngineCore at 6,047 MiB
  and 3% utilization. The replacement receipt is independently rehashable and
  supersedes an initially non-rehashable in-memory-Buffer serialization. Two
  other fail-closed development attempts are retained: an
  uninstantiated pre-request Prometheus route series is now correctly treated
  as zero while its metric family remains required, and measured first-CUDA
  cold/warm drift is explicitly bounded while warmed repeats remain exact.
- All 480 primary chunks were embedded in 15 durable batches and persisted
  with complete input, request, response, operation, batch, model, run, and raw
  artifact provenance. Final receipt:
  `bca6706cf683f265f61bff8e7f7abe0531bf8722afb36f479d29c66f7feca1aa`.
  Generation deltas are exactly 15 HTTP requests, 480 successes, 23,736 prompt
  tokens, 480 latency observations, zero errors, and zero preemptions; request
  p50/p95 is 36.843/41.932 ms. Ordered input, service-vector, and SQL-storage
  hashes are `eb3559f43fa413bc70d514924c57ea044b25ab2a0c1e97ee91593144216099fc`,
  `012daefc3106442687f0842bf442e3ec1f37c1558e6f07dd354165034af54aa5`,
  and `f8a929f0bc3f87ddf8a2edc4fe6406e52900da7f2759d2169800188817cf3d31`.
  Across 491,520 components, SQL float32 conversion has worst absolute drift
  `4.995651239902976e-9` and minimum cosine `0.9999999999999969`.
- App-owned lexical full-text, exact-vector, and hybrid RRF retrieval now
  persist complete query-vector/component-rank/result evidence while retaining
  least privilege. The 11-query x three-mode development gate passed with
  receipt `f6a3b4aefbca5214beee3272566f5f01d696f4be3dcb922a9660db2ea86d1dfd`:
  all modes reached recall@5=1 and MRR=1, with lexical/vector/hybrid SQL p50 of
  14.074/72.502/75.022 ms. These deliberately easy canaries prove mechanics,
  not hybrid lift; that claim is reserved for held-out packet evaluation.
- The bounded multi-turn agent gate passes eight terminal routes with receipt
  `8449a07300d761fdac156962f8c9b7056f48f8bd5c96d825745b82ce3ddc10b8`:
  17 model turns, 13 tool calls, three safely persisted decisions, exact cache
  reuse, a deterministic snapshot miss, four rejection classes, zero executed
  actions, zero retained leases, and closed/link-complete traces. Prompt/model/
  retry/tool/validation/decision phases persist in both hash-chained files and
  SQL; raw requests, responses, and full tool results are independently
  rehashed. Two uncached hybrid searches produced exactly two real CUDA
  embedding requests/successes, 17 prompt tokens, two latency observations,
  zero errors, and zero preemptions. The primary transport is user-turn-only
  and unconstrained: no system message, OpenAI `tools`, or `response_format`.
  Queue migrations 029/030 add RCSI-safe locking reads and normalize pooled
  session isolation; three fail-closed attempts and their evidence impact are
  append-only in `EXPERIMENT_LOG.md`.
- The target chat lifecycle and port gate are implemented but have not loaded
  any chat weights. They pin the dynamic vLLM container identity and retain
  raw transport, service logs, complete requested/effective decode settings,
  model metrics, GPU state, sequential/batched canaries, finish/repair/token
  fields, and hash-chained telemetry. Authored migration 031 will preserve a
  server-split reasoning channel independently from final content and raw
  envelopes, attribute service samples to instance/phase, and store
  evaluator-only held-out retrieval rows; it remains unapplied until capture
  is no longer mutating SQL.
- The executable Tier 1 control policy was hash-frozen during capture and
  before packet construction or target-output inspection. Error-number and
  exact-signature masking now transforms both packets and tool results;
  shuffled retrieval clones evaluator-only wrong-runbook rows into separately
  provenance-marked agent-visible retrieval runs; and the sequential 48-cell
  batching control preserves distinct job/raw/telemetry identities. Every
  control prediction links to its primary A-tools source, scores separately,
  and materializes exact raw-response, decision, ordered-tool-call, and result
  agreement. The timing and its no-data-impact limitation are disclosed in the
  preregistration and experiment log. Durable commit: `764c9c3`.
- The unfrozen `logwarden-standard-v1` scenario catalog now contains 60
  group-isolated templates and 600 deterministic variants across all ten
  incident families and all five regimes. Exact role allocation is 60 dev, 60
  calibration, 300 test-ID, 120 test-variant-holdout, and 60 test-unknown;
  every family has at least 40 held-out ID/variant episodes and no group crosses
  a split. Catalog, manifest, schedule, and structural-gate hashes are
  `5bc260b8c11c6ace97cb5f047dd0ad7f68958860cc24c3e0e43313ee9f05934f`,
  `50d47873fda8ccd3367ddbd5f9f1356f1fd306f6406c33de122471a1c5e7872d`,
  `3b7097f5b855d91741bc2d470e3074040464b96a9ae66e1df766639c6daea20b`,
  and `5535d9f8c29d58d6b1ec222b3ab0842c20de0c669d01367d3118ff07e1f17f82`.
  Schedule `standard-v1` has 600 items and a final planned offset of 8,039,000
  ms; at the 09:13 UTC check it had durably injected 483 episodes with zero
  injector errors. Verification is intentionally finalized after capture.
- Long-run preflight fixed the injector's schedule-clock resume semantics
  before any standard row existed. A resumed process now reconstructs the
  original clock from the earliest durable execution rather than waiting the
  full planned duration again; the terminal receipt records the reconstructed
  origin and resume state. This changes no scientific input or captured data.
- All eight real incident injector families passed the three-signal
  development feasibility gate: 8/8 capture verification receipt
  `0ff81209e857278a56c9920c233281cb34d4b97815d60936efff9291b407e79a`
  and 32-check deep receipt
  `8ace73c31d9b07e7dba3388900eca9b0da94d71740176313f76b260f901cabe3`.
  The gate proves exact raw-token attribution, three observed driver signals,
  current deadlock semantic hashes, sustained blocking/active transactions,
  >=90% log utilization with `ACTIVE_TRANSACTION`, snapshot hashes, cleanup,
  and zero XE loss counters. A v1 development run was rejected when the deep
  gate exposed proximity-only attribution of connection setup batches; the
  corrected verifier requires intrinsic full-token RPC/batch evidence and
  deterministically rebuilds links. No standard data existed, so result impact
  is limited to stronger pre-capture independence.
- SQL Server rejected the requested `NO_EVENT_LOSS` mode for this
  `error_reported` event mix (error 25643). The documented capability-forced
  contract uses `ALLOW_SINGLE_EVENT_LOSS`, 1-second dispatch, 16 MiB x 20
  files, and fail-closed zero loss counters. Asset hash:
  `667bee8af19d8b2dba416cd31968215945b8f39ee1121e6294e37549bc2068c1`.
- Current doctor snapshot:
  `2a6f74acb8e0c1a35c06faa437e3565b13df1be26d934823a84ca50bd6466548`
  (`PASS`); SQL integration receipt:
  `407d3147ac4fe8898475928f7debb83e878bb0f0500a267ac579192db31ad3b2`
  (8/8); least-privilege tool receipt:
  `fca0eaa2e865c00082141f98527c85b9e6f83f77caeae10a2254c0f4a879126f`
  (9 positive/11 negative).
- The user-supplied Mac/Foundry/report series and follow-ups through `ecec5f4`
  are
  merged. Their platform-separated Foundry/MLX registry, Apple Silicon
  Docker/Rosetta setup, 16-episode dev harness, six-model retained results, and
  report generator coexist with the Colab path. Colab Compose validation,
  doctor, SQL, security, backup/restore, and all pre-merge tests pass; the
  committed nine-serving x 16-episode HTML report rebuilds byte-for-byte from
  retained data and its template after correcting stale eight-serving
  cardinality. Foundry is absent on this Linux VM and, by user direction,
  remains a post-governed-model validation lane with non-comparable results.
- Active milestone: complete standard capture, then build/audit its packets,
  assess held-out retrieval, and freeze the corpus. The qwen-smoke chat service
  may load only for its
  separate real `/metrics` gate; no long chat campaign may start until that
  gate passes. Foundry model validation is deliberately deferred until after
  the governed local models.

## Fresh-VM preflight

Observed at `2026-08-23T02:40:15Z` from the launch shell:

- host/kernel: Linux `6.6.122+`, x86-64
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- GPU UUID: `GPU-e07963ac-e6be-8913-3294-b83917338161`
- driver: `580.82.07`
- VRAM: 97,887 MiB total; 97,251 MiB free
- local filesystem: 236 GiB total; 188 GiB free
- mounted Drive filesystem: present; 178 GiB free
- Node.js: `v20.19.0`
- npm: `10.8.2`
- Python: `3.13.15`
- GitHub SSH read/write authentication: passed (`git ls-remote` and branch
  push)
- Docker: Engine 29.7.2, Compose 5.5.0, rootless `fuse-overlayfs`
- NVIDIA Container Toolkit: 1.20.0; generated CDI spec
- GPU/service ownership: Lab 3 rootless Docker daemon active; Qwen embedding
  vLLM/EngineCore is resident on port 8011; no chat vLLM, generator, watchdog,
  or Lab 2 service process is present
- SQL Server: 2025 RTM-CU8 `17.0.4075.5`, healthy on reserved port 1434;
  exact FTS package `17.0.4075.5-1`, `IsFullTextInstalled=1`; derived image
  `sha256:eb4ee252ae0ff6a18b5b40e05eea283e251ee69fbe8926e28def6d0adb43f95f`

The real CUDA allocation/synchronization hard gate and Qwen embedding model
port gate passed. A separate chat-model port gate remains required before
chat-model-scale work.

## Planned isolated resources

These are reserved for Lab 3 and may be changed only with a recorded update:

- Compose project: `aidataapps-logwarden`
- SQL host port: `1434`
- chat-model host port: `8010`
- Qwen embedding host port: `8011`
- optional BGE embedding host port: `8012`
- databases: `LogWardenControl` and disposable `LogWardenWorkload`
- local runs: `aidataapps/logwarden/runs/<run-id>`
- Drive runs: `/content/drive/MyDrive/aidataapps/lab03/runs/<run-id>`

The inherited Lab 1/2 directories and their branches are read-only inputs.

## Active processes and checkpoints

<!-- lab3-watchdog-status:start -->
- Last watchdog checkpoint: 2026-08-23T17:22:11.323Z
- Last watchdog Git head: `11fa7de535a49943f41e2033d52f7a01e6a83b03` on `aidataapps-logwarden`
- Last watchdog disposition: clean source checkpoint
- Last watchdog database receipt: `d9511011908a4f13ae6641fc065f8165aa272bdd7830a59ed5f44ee56d08309d`
- Last watchdog run: `logwarden-smoke-20260823T031714Z`
<!-- lab3-watchdog-status:end -->

- Long-running scientific process: no replay is active; Gemma calibration is
  closed before test access and protected primary replay is now authorized.
- Infrastructure process: rootless Docker is supervised by retained Codex exec
  cell `64558`; detached children are reaped in this environment
- Telemetry: continuous `gemma-4-31b-residency` whole-system sampler is retained
  in exec session `65883` (node PID 595355), epoch
  `bde528a1-bb5c-4b66-91b5-ac19591487dd`; it includes download, load, compile,
  the cold failure, retry, gates, and subsequent inference
- Watchdog: recurring bounded 20-minute backup/Git-bundle/push/run-mirror
  supervisor is retained in exec session `11505` (bash PID 589796); its first
  complete bounded cycle passed and temporary SQL staging cleanup was verified
- SQL backup: both databases passed COPY_ONLY/CHECKSUM backup, VERIFYONLY,
  full disposable restore, physical CHECKDB, and teardown; frozen checkpoint
  hash `20a468ccbb80f5ada685f74beade3a0919655fc5c8bfe914f539514aa6e9fd06`
- Active run ID: `logwarden-smoke-20260823T031714Z`
- Capability snapshot: `2a6f74acb8e0c1a35c06faa437e3565b13df1be26d934823a84ca50bd6466548` (`PASS`)
- SQL integration receipt: `407d3147ac4fe8898475928f7debb83e878bb0f0500a267ac579192db31ad3b2` (8/8 passed)
- Last durable implementation checkpoint: `614546e` (labeled runtime-only GPU
  memory override with model profile/hash unchanged; all 140 tests pass)
- Last durable Drive checkpoint: this file

Before the first job expected to exceed 20 minutes, launch the tested
Lab-3-scoped checkpoint watchdog. It pushes committed work, makes a verified
branch bundle and worktree recovery capture, backs up both Lab 3 databases,
archives/mirrors only the active Lab 3 run, and updates this file without
touching Lab 2 state.

## Exact resume procedure

```bash
cd /content/worktrees/aidataapps-logwarden
git fetch origin
git pull --ff-only
git status --short --branch
cat aidataapps/resume.md
cat aidataapps/inprogress_lab3.md
ps -eo pid,ppid,etimes,args | rg 'logwarden|replay-run|vllm serve|EngineCore|checkpoint-watchdog' || true
nvidia-smi
```

Then inspect the newest `EXPERIMENT_LOG.md`, active run pointer, watchdog log,
SQL job/work-item state, current Muse port-gate receipt, and Drive checkpoint
before launching anything. The campaign is frozen/running and Muse is
complete through calibration, protected primary, deterministic router,
protected scoring, all three controls, telemetry reconciliation, and its pinned
database checkpoint. Resume by stopping/evicting Muse and starting Gemma as
shown in the authoritative state section. Do not start another chat model while
the Muse container is resident.
If `docker info` fails,
rerun `./scripts/colab-host-init.sh` or launch the rootless daemon in a retained
cell.

## Completion rule currently in force

Tier 1 must close before Tier 2 begins: `repro.sh --mode rows` must pass from a
fresh checkout; `SCORECARD.md` must regenerate byte-for-byte; the Tier 1 safety
audit must be clean; every Tier 1 job must have a terminal disposition; and all
deferred or adapted requirements must be listed with their result impact.
