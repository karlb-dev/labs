# Validation Record

This file distinguishes checks that run in any Node environment from checks
that require the outer GPU VM. Update it with exact commands and artifacts when
the full stack is run.

## Required Gates

| Gate | Command | Evidence |
| --- | --- | --- |
| TypeScript build and CPU unit tests | `npm run check` | terminal result and commit |
| SQL Server 2025 startup/version | `./scripts/env-init.sh` | `@@VERSION` output |
| native vector seed/search | `npm run db:setup` | row counts and API citations |
| vLLM embedding and chat readiness | `GET /health`, `GET /v1/models` | `/ready` pass |
| answer + executed-action smoke | `npm run smoke` | cited chunks and work-order ID |
| model matrix | `./scripts/benchmark-matrix.sh` | ignored `runs/benchmark-*` artifacts |

## Initial Generation

Generation environment: 2026-08-22, branch `aidataapps-rag`.

| Check | Result |
| --- | --- |
| `npm run check` on Node 20.19.0 | PASS — TypeScript build; 4 files / 11 tests |
| `npm audit --audit-level=high` | PASS — 0 vulnerabilities |
| shell syntax for all `.sh` scripts | PASS |
| JSON parse for registry, seed data, cases, package, lockfile | PASS |
| Compose parse for `compose.yaml` plus `compose.colab.yaml` | PASS |
| model registry/list CLI | PASS — all six profiles resolve; four-cell matrix order is stable |
| rootless Docker plus Compose | PASS — Docker CE 29.7.2; Compose 5.5.0 |
| NVIDIA CDI passthrough | PASS — RTX PRO 6000 Blackwell Server Edition, driver 580.82.07, 97,887 MiB |
| NVIDIA Container Toolkit | PASS — 1.20.0 |
| SQL Server container | PASS — SQL Server 2025 RTM CU8, 17.0.4075.5, Developer Edition |
| database seed | PASS — 5 assets, 6 documents, 14 native-vector chunks |
| embedding service | PASS — `Qwen/Qwen3-Embedding-0.6B`, 1024 dimensions |
| `npm run smoke` with `qwen-smoke` | PASS — grounded answer, citation, and work order ID 1 |
| `npm run smoke` with `qwen-3.8-27b` | PASS — grounded answer, citation, and work order ID 2 |

The Codex process is root inside Colab's outer container but does not have
`CAP_NET_ADMIN`; a conventional rootful daemon therefore cannot create fresh
network, proc, or cgroup namespaces. Validation used the checked-in rootless
bootstrap, shared host namespaces, `fuse-overlayfs`, and NVIDIA CDI. GPU access
was verified from a nested container before any model benchmark was accepted.

## Full GPU Matrix

All rows used one repeat of the same four fixed application-contract cases on
the seeded SQL database. Latencies include retrieval, generation, output
validation, and action proposal. The ignored run directories contain
`run.json`, `results.jsonl`, `summary.json`, and `summary.md`.

| Profile | Passed | p50 | p95 | Artifact |
| --- | ---: | ---: | ---: | --- |
| `muse-glimmer-30b` | 4/4 | 7,517 ms | 9,745 ms | `runs/benchmark-muse-glimmer-30b-20260822T195125Z/` |
| `gemma-4-31b` | 4/4 | 4,234 ms | 4,623 ms | `runs/benchmark-gemma-4-31b-20260822T200252Z/` |
| `olmo-3.1-32b-instruct` | 4/4 | 4,016 ms | 5,155 ms | `runs/benchmark-olmo-3.1-32b-instruct-20260822T200922Z/` |
| `qwen-3.8-27b` | 4/4 | 6,312 ms | 25,181 ms | `runs/benchmark-qwen-3.8-27b-20260822T201705Z/` |

Gemma 4 hit the confirmed vLLM 0.27.1 heterogeneous-`head_dim` regression,
so that profile is pinned to vLLM 0.26.0 and passed in text-only mode. Muse
also runs text-only for this RAG workload and uses its dedicated image/parser;
`reasoning_strength=low` was needed to keep action JSON within the response
budget. Qwen 3.8 runs text-only to disable unused multimodal inputs. The Qwen
p95 reflects its coldest case in this single-repeat exploratory run and is not
a general throughput claim.

Large weights were removed one profile at a time after their artifacts were
recorded to fit the ephemeral VM disk. The final Qwen 3.8 weights and server
were left resident; deleted weights are reproducible from the pinned Hugging
Face revisions.
