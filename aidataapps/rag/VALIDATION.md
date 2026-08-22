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
| YAML parse for `compose.yaml` | PASS |
| model registry/list CLI | PASS — all six profiles resolve; four-cell matrix order is stable |
| `./scripts/env-init.sh --model qwen-smoke` | BLOCKED before mutation — Docker unavailable in the execution shell |

The generation shell is an unprivileged inner container rather than the outer
GPU VM. It exposes the host's NVIDIA 580.82.07 kernel-module metadata under
`/proc/driver/nvidia`, but has no Docker CLI or socket, no `/dev/nvidia0`, an
empty capability bounding set, and read-only user-namespace maps. Installing a
Docker client inside it would not provide a daemon or GPU device, so SQL Server
and vLLM were not falsely marked as validated.

The checked-in unit suite is deliberately independent of Docker, SQL Server,
model weights, and GPU availability. Full-stack validation remains open until
`./scripts/env-init.sh` is run from the outer VM host (or from a development
container that receives the Docker socket and NVIDIA devices).
