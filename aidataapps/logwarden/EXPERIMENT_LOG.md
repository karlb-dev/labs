# LogWarden Experiment Log

Append-only operator record. Times are UTC. Entries are written before an
adaptation can affect evidence whenever possible; superseded facts remain in
place and receive a later disposition.

- 2026-08-23T02:40:15Z — LW-0 began on a clean Colab VM. Created dedicated
  worktree `/content/worktrees/aidataapps-logwarden` and branch
  `aidataapps-logwarden` from exact Lab 2 head `88ea443092cd27226272776e6c6fcf8fbce329de`.
- 2026-08-23T02:40:15Z — Preflight found an idle NVIDIA RTX PRO 6000
  Blackwell Server Edition (97,887 MiB total; 97,251 MiB free), 188 GiB local
  free, mounted Drive with 178 GiB free, and no Docker installation or active
  Lab 2/model process on this VM.
- 2026-08-23T02:47:00Z — GPU hard gate passed with PyTorch 2.11.0+cu128: a
  real CUDA tensor allocation, reduction, host readback, and synchronization
  completed on the RTX PRO 6000. No CPU fallback is permitted for model-scale
  stages.
- 2026-08-23T02:48:00Z — Implementation sequencing clarification (pre-data):
  the executable bootstrap uses `run:init -> db:setup -> doctor` after service
  startup, because the full capability doctor persists its evidence into the
  control database. This changes only command order, not any scientific input
  or claim. A bounded host/service preflight still runs before database setup.
- 2026-08-23T02:49:00Z — Current official method inputs were checked before
  implementation: SQL Server 2025 Linux full-text installation, Extended
  Events dispatch/session syntax, Query Store configuration, and vLLM JSON
  schema response handling. Runtime capability probes remain authoritative.
- 2026-08-23T02:55:00Z — User-directed pre-data scope adjustment: promote the
  practical observability layer ahead of every long LLM run, even where the
  addendum labels high-volume telemetry tables Tier 2. The gate now requires
  append-only file journaling plus incremental SQL records for the full agent
  state machine, raw vLLM metrics snapshots and discovered metric names,
  client monotonic request timings, queue/XE/SQL/GPU samples, retry and repair
  paths, and telemetry completeness/reconciliation tests. This does not alter
  packets, prompts, labels, arms, or scoring. It increases write/sampling
  overhead, so the development tier must measure that overhead and freeze the
  sampling interval before the primary campaign.
- 2026-08-23T03:00:34Z — Inspected the PI-supplied `cloud-deploy-agent`
  reference at exact commit `273221e97e36abea34e47961d4deb42455750f02`.
  Adopted reconstructable semantic projections over one immutable journal,
  request-level client/server correlation, per-call prompt/tool/config hashes,
  explicit loop-limit and response-failure classes, safe forced terminal
  dispositions, critical-path versus additive timing, and per-job/grouped
  diagnostics. Rejected TPU-specific fields and heuristic token/timing values
  as observations: LogWarden will retain their provenance or null. This is an
  instrumentation-only pre-data change and does not affect episode assignment,
  prompts, arm definitions, labels, or primary outcomes; it adds storage and
  measured overhead that the development gate will quantify.
