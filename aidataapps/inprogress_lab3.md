# Lab 03 complete — LogWarden

Last updated: 2026-08-23 22:06 UTC

Read `/content/drive/MyDrive/aidataapps/resume.md` first for shared recovery
rules. This is the authoritative Lab 3 handoff.

## Location and ownership

- Worktree: `/content/worktrees/aidataapps-logwarden`
- Branch/remote: `aidataapps-logwarden` / `origin/aidataapps-logwarden`
- Lab: `/content/worktrees/aidataapps-logwarden/aidataapps/logwarden`
- Run: `runs/logwarden-smoke-20260823T031714Z`
- Drive mirror:
  `/content/drive/MyDrive/aidataapps/lab03/runs/logwarden-smoke-20260823T031714Z`
- Final mirrored artifact-inventory SHA-256:
  `951bf7c836ab7212588f03cd011dacf4c0d13e25bb6fa8428a88c0e51e2c5c03`
- Source plan:
  `/content/drive/MyDrive/aidataapps/lab03/aidataapps_logwarden_lab_3_spec.md`
- Governing addendum:
  `/content/drive/MyDrive/aidataapps/lab03/aidataapps_logwarden_lab_3_spec_addendum.md`
- Branch origin: created from the Lab 2 head at
  `88ea443092cd27226272776e6c6fcf8fbce329de`; later synchronized with the
  completed Lab 2 mainline through merge commit `47ebe2e`.
- Durable Tier-1 report implementation: `bc58971`. The final closeout changes
  are the current branch head; use `git rev-parse HEAD` after pulling.

## Completion state

Lab 3 Tier 1 is complete. No model campaign, watchdog, analysis job, or
reproduction container remains active. The primary adjudication is
`CLEAN_NULL`, not a positive model result.

The embedding service was stopped cleanly after closeout, so the GPU is free
and no vLLM process remains. SQL Server is intentionally left healthy on port
1434 for inspection; all model/cache volumes and database backups are retained.

- Frozen campaign hash:
  `e105cfdd5af5345464853406c8d707018232f3326c909ca476970fb7137cbcf6`
- Freeze receipt:
  `1819a96b69cb90c73c450aa2da5376bf6ab565c3d20f9293192746e5fc4e60df`
- Corpus: 600/600 capture gates passed, 600 packets, zero leakage findings.
- Three governed chat profiles completed every authorized Tier-1 cell: Muse
  Glimmer 30B, Gemma 4 31B, and Qwen 3.8 27B.
- OLMo 3.1 32B is formally `STOP_PORT` after two independent hash-valid gates.
  Its unconstrained output consistently omitted required contract fields.
  A separate guided-decoding diagnostic passed but is retained only as Tier-2
  portability evidence and was never pooled into Tier 1.
- Final analysis: 8,430 predictions, 5,085 metric rows, 36 paired contrasts,
  10,000 bootstrap draws and 1,000 within-stratum permutations; receipt
  `7fcfb31517c560a7fdaad9b03b800dec10898415c831377e3407135fff41737d`.
- No contrast passed both preregistered gates (familywise CI lower bound above
  zero and Holm-adjusted p <= 0.05).
- B1 rules remained the strongest executable arm: action accuracy `0.9167`,
  end-to-end success `0.3333`, cost-weighted loss `1.1667`. B3 is an
  evaluator-only oracle ceiling, not a deployable comparator.
- The carry-forward architecture is deterministic rules for the known head,
  explicitly gated retrieval/model assistance for residuals, and SQL Server as
  the durable evidence, queue, and evaluation plane—not autonomous remediation.

## Performance and observability state

The user-requested pre-inference observability gate was implemented before long
GPU runs. Telemetry is dual-written to append-only files and SQL, reconciled,
and included in the deterministic performance report.

- Reconciled evidence: 11,332 closed traces, 65,321 closed spans, 9,986 paired
  model requests/responses, 4,161,680 unique metric samples, and 16,501 raw
  snapshots.
- Final telemetry receipt:
  `8c2de36fe09cdc0905cfa54977b7e32aeb49641360fdc4458e698ad0e7f3cb84`.
- Captured surfaces include agent/phase p50/p90/p95/p99 latency, request and
  token counts, finish/error/preemption outcomes, queue depth/age, tool and
  validation outcomes, SQL resource samples, vLLM counters/concurrency, and
  GPU utilization/memory/power/temperature.
- Metrics not exposed by the retained interfaces are labeled unavailable,
  never inferred from proxies: true TTFT/inter-token latency, vLLM throughput
  gauges, KV occupancy, SQL process CPU/log-used percentage, and Tier-1 Query
  Store interval rows.
- Performance report:
  `aidataapps/logwarden/LOGWARDEN_PERFORMANCE_REPORT.md` and
  `runs/logwarden-smoke-20260823T031714Z/reports/PERFORMANCE_REPORT.md`.

## Results, persistence, and reproducibility

- Self-contained interactive comparison report:
  `runs/logwarden-smoke-20260823T031714Z/reports/LOGWARDEN_TIER1_REPORT.html`;
  companion deterministic payload:
  `runs/logwarden-smoke-20260823T031714Z/reports/LOGWARDEN_TIER1_REPORT_DATA.json`.
  A convenience copy is also at
  `/content/drive/MyDrive/aidataapps/lab03/LOGWARDEN_TIER1_REPORT.html` with
  `/content/drive/MyDrive/aidataapps/lab03/LOGWARDEN_TIER1_REPORT_DATA.json`.
  Payload receipt:
  `b49a5b3860b54782d93cdb76a0ae24ed3c611fd382a3e929b9d208335e3de61b`.
  The generator verifies all frozen scientific inputs and receipts before
  publication; adding this post-closeout presentation did not mutate the
  frozen campaign or final SQL boundary.

- Report input: 85,596 rows from 17 deterministic SQL queries plus 33 signed
  evidence artifacts; manifest receipt
  `02835d283cb931566633e9551ddafd248d116348c65b1909d009fe0db32977be`;
  query bundle
  `5774657af144896634eba445a78991531eeeab12635e5f75b1e7753ee6337fce`.
- Deterministic outputs: 22 scorecard rows, 206 taxonomy rows, 41 claims, and
  all 13 required figure-source CSVs; output receipt
  `ee357e34d30b3195cfd20b1a7c905d746ca17514d515757e0656b769a8be4b8f`.
- SQL persistence: 206 taxonomy rows, 41 claims, and 15 report snapshots;
  receipt
  `38c3d2950939f5271a208d3adf97a3e4977b3ce57b99378c81ef50171c8a1823`.
  An immediate dry run inserted zero rows and verified idempotence.
- Row-only reconstruction passed all 42 test files / 148 tests, verified every
  retained input hash, and rebuilt claim-bearing outputs byte-for-byte. Stable
  receipt:
  `cfddf486ec4fea5e94ff1cf47f059f5f23226f22d48bc5133c6a89e61f7e313d`.
- Fresh-SQL reconstruction restored into isolated database names, passed
  physical `CHECKDB`, found 82 control tables / 32 migrations and five workload
  tables, reproduced the exact row manifest, and rebuilt reports without
  drift. Stable receipt:
  `843c0c7efa97f02bb2dd99300eaa663b17ed4e90708e476dd07d213af4c5df0e`.
- Latest post-ledger restore execution:
  `d31e14fe95a52660cd63f35f1e57be28985e8e831f1ac68499ecb84a10ad9f2f`;
  database receipt
  `aafd5cbeb857bec5e3970cd1a7f199491e045c1834ef1279399ddae6efb349f2`.
  It explicitly verified 206 taxonomy rows, 41 claims, and 15 report snapshots.
- Final post-ledger backup receipt:
  `d8316dd88e172f9951f55a3afc0443265d1b6c53efd4b192c14b68f87c27eac1`.
  Control backup: 1,047,756,800 bytes,
  `278c17722df0106d8b7bc390832e950da5b07708347ae4144a4da6a74ebf251a`;
  workload backup: 774,144 bytes,
  `53481e310b6cb66a229d3e7d258cd45aa20d3484cdda30b86e0c9067dce70568`.
- Final checkpoint pin/retention receipt:
  `b5fcae06f2297ddf0cb2ed6398835dfee2cc432ba6db5d4ba6d177901b3e9604`.
  It retained the freeze, each profile boundary, the Tier-1 final boundary, two
  rolling points, and two recovery bundles while deleting only six redundant
  local/Drive files (2,096,950,416 bytes).

## Runtime adaptations and scientific impact

Every adjustment is append-only in `EXPERIMENT_LOG.md`. Important closeout
items:

- Gemma needed a runtime-only GPU residency cap change from 0.78 to 0.79 to
  expose enough KV cache for the frozen 16K request. Prompts, weights, context,
  decode, and model/profile hashes did not change; the +1 point is disclosed in
  performance results.
- Missing deterministic B0/B2/B3 prediction rows were derived before final
  analysis from already-frozen definitions. No LLM was called and no model row
  changed; this completed the declared baseline grid.
- Colab lacks `ensurepip`; report generation verifies and uses the host
  scientific stack when an isolated venv cannot be constructed.
- Rootless nested Docker cannot use `docker cp` with the required read-only
  `/proc` bind or create a private PID namespace. Restore mode therefore uses a
  hash-verified, short-lived read-only backup bind and host PID mode.
- Rootless SQL teardown can respawn once before Docker owns removal. The final
  cleanup concurrently removes the exact Compose container and drains only
  processes matching its random ephemeral SQL port and exact SQL executable.
  The isolated container, volumes, network, staging copies, and shims were
  verified absent; primary port 1434 was never touched. This affects teardown
  reliability only, not scientific evidence.

## Foundry and Mac portability

The user branch with Foundry, Mac, and post-run report support is merged. Linux
Foundry was not installed on this x86_64 Colab VM because the governed profile
defines Foundry Local as a distinct Apple-Silicon/Mac evidence plane, not a
substitute for digest-pinned CUDA/vLLM inference.

Existing M4 Max evidence already verifies Foundry Local 0.10.3, a warm
`qwen3-4b` decision canary at about 60 tokens/s, 1024-dimensional Qwen
embeddings, and multi-model Foundry/MLX evaluation. See `docs/MAC_PROFILE.md`
and `docs/reports/mac-eval-bench-20260823.html`. Those rows remain explicitly
non-comparable to and excluded from the frozen Tier-1 campaign.

## Resume / audit commands

No long job should be resumed. To audit the frozen result:

```bash
cd /content/worktrees/aidataapps-logwarden/aidataapps/logwarden
source scripts/runtime-env.sh
npm ci
npm run check
./repro.sh --mode rows --run runs/logwarden-smoke-20260823T031714Z
./repro.sh --mode restore --run runs/logwarden-smoke-20260823T031714Z
```

Start from these files:

- `runs/logwarden-smoke-20260823T031714Z/reports/LOGWARDEN_TIER1_REPORT.html`
- `runs/logwarden-smoke-20260823T031714Z/reports/LOGWARDEN_STATE_OF_RECORD.md`
- `runs/logwarden-smoke-20260823T031714Z/reports/SCORECARD.md`
- `runs/logwarden-smoke-20260823T031714Z/reports/VALIDATION.md`
- `runs/logwarden-smoke-20260823T031714Z/reports/LOGWARDEN_REPRODUCIBILITY.md`
- `runs/logwarden-smoke-20260823T031714Z/reports/LOGWARDEN_LIMITATIONS.md`
- `EXPERIMENT_LOG.md`

Tier 2/3 items remain intentionally unclaimed: live replay/throughput storms,
raw-event correlation, more guided-decoding cells, a second embedding profile,
Query Store interval analysis, temporal incidents, expanded SQL security and
Audit, SQL-native chunking/embedding/ANN comparators, BACPAC, columnstore/JSON
migrations, additional operational reports, and UI.
