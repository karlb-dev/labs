# Lab 1: RAG With SQL Server and Local Models

This is the initial code asset for an app-shaped RAG lab. It combines a
TypeScript API, SQL Server 2025 native vectors, an OpenAI-compatible local vLLM
server, and one deliberately narrow action. The fictional domain is Northstar
Bikes, a shared e-bike service desk with maintenance manuals, safety policies,
assets, and work orders.

The first question is not “which framework can hide the most plumbing?” It is:

> What evidence crosses each boundary when retrieval, generation, and an
> operational action share one application?

## What Runs

```text
POST /api/query
  -> Qwen3 embedding service (vLLM)
  -> exact VECTOR_DISTANCE search (SQL Server 2025)
  -> selected chat model (vLLM)
  -> validated answer or create_work_order proposal
  -> optional, explicitly enabled SQL transaction
```

The seed database contains five fleet assets, six source documents, fourteen
grounded chunks, native 1024-dimensional vectors, and an initially empty work
order table. Exact search is intentional: it gives later labs a trustworthy
baseline before SQL Server's preview approximate vector index is introduced.

## Primary Target

- Ubuntu 22.04 x86-64
- one NVIDIA RTX PRO 6000 Blackwell (96 GB)
- NVIDIA Release 580 driver or newer
- Docker Engine with Compose and NVIDIA Container Toolkit
- Node.js 20 or newer

Verify the host before the first run:

```bash
docker --version
docker compose version
nvidia-smi
docker run --rm --gpus all ubuntu nvidia-smi
```

The SQL image is Microsoft's official
[`mcr.microsoft.com/mssql/server:2025-latest`](https://mcr.microsoft.com/product/mssql/server/about).
The GPU services use the official
[`vllm/vllm-openai`](https://docs.vllm.ai/en/stable/deployment/docker/) images.

## Quick Start

The environment initializer creates `.env` with a generated SQL password,
installs Node dependencies, pulls and starts SQL Server plus the embedding
model, starts one chat profile, creates and seeds `RagLab`, builds/tests the
app, and runs a real answer-plus-action smoke test.

```bash
cd aidataapps/rag
./scripts/env-init.sh
```

First startup downloads the vLLM images and model weights, so it can take much
longer than later runs. The default `qwen-smoke` profile is a 4B plumbing test.
To initialize directly with a large profile:

```bash
./scripts/env-init.sh --model olmo-3.1-32b-instruct
```

Gemma and any other access-controlled model require a Hugging Face token in
`.env` after accepting the model's license:

```dotenv
HF_TOKEN=hf_...
```

Run the API after initialization:

```bash
npm run start
```

Ask a grounded question:

```bash
curl http://127.0.0.1:3000/api/query \
  -H 'content-type: application/json' \
  -d '{"query":"What should I inspect when a Comet S2 loses motor assistance after rain?"}'
```

Actions are dry-run proposals unless the caller opts in:

```bash
curl http://127.0.0.1:3000/api/query \
  -H 'content-type: application/json' \
  -d '{
    "query":"Create a high-priority work order for NB-104 to inspect intermittent motor loss after rain.",
    "allowActions":true
  }'
```

Stop the containers without deleting SQL data or downloaded model caches:

```bash
./scripts/env-down.sh
```

## Model Profiles

Profiles live in [`config/models.json`](config/models.json). Model identity,
revision, prompt compatibility, vLLM image, and engine arguments are part of
the recorded configuration.

| Profile | Exact model | Purpose |
| --- | --- | --- |
| `qwen-smoke` | `Qwen/Qwen3-4B-Instruct-2507` | cheap end-to-end plumbing |
| `muse-glimmer-30b` | `meta-models/Muse-Glimmer-30B` | 30B agentic/multimodal comparison; dedicated Muse parsers |
| `gemma-4-31b` | `google/gemma-4-31B-it` | 31B comparison; system instructions folded into the user turn |
| `olmo-3.1-32b-instruct` | `allenai/Olmo-3.1-32B-Instruct` | repo-aligned 32B instruction spine |
| `qwen-3.8-27b` | `Qwen/Qwen3.8-27B` | current 27B Qwen comparison cell |
| `qwen-3.6-27b-pinned` | `Qwen/Qwen3.6-27B` | historical repo-pin replay |

List, switch, inspect, or stop the chat server independently:

```bash
npm run model -- list
npm run model -- start --profile qwen-3.8-27b --replace
npm run model -- status
npm run model -- stop
```

Only one chat profile is resident at a time. The embedding service stays
resident so every chat model is measured against the same stored vectors.
Profiles cap context at 16K for this RAG experiment rather than spending most
of the 96 GB device on an unused long-context KV cache.

## Benchmarking

The benchmark checks four fixed cases: three grounded answers and one dry-run
action proposal. It records pass/fail, citations, latency, Git commit, exact
model revision, vLLM container image digest, Node version, and GPU diagnostics.

Benchmark the currently running profile:

```bash
MODEL_PROFILE=qwen-3.8-27b npm run benchmark -- \
  --profile qwen-3.8-27b \
  --repeats 3
```

Run the default serious-model matrix sequentially:

```bash
./scripts/benchmark-matrix.sh
```

Or choose a subset:

```bash
./scripts/benchmark-matrix.sh olmo-3.1-32b-instruct qwen-3.8-27b
```

Each run writes:

```text
runs/benchmark-<profile>-<timestamp>/
  run.json
  results.jsonl
  summary.json
  summary.md
```

This is an application-contract benchmark, not a general model leaderboard.
Its claims are limited to retrieval grounding, output-shape compliance, action
selection, and latency on the fixed seed corpus.

## Development Without the Full Stack

Unit tests inject fake repositories and inference gateways, so API and action
policy work can be debugged without a GPU or SQL Server:

```bash
npm ci
npm run check
```

Useful full-stack commands:

```bash
npm run db:setup   # reset and reseed the sample tables
npm run smoke      # requires a running app and executes one test work order
npm run dev        # watch the TypeScript server
```

## Action Boundary

`create_work_order` is the only allowlisted action. The model can propose it,
but the application validates the JSON shape, rejects invented citations,
requires `allowActions: true`, verifies that the asset exists inside the same
SQL transaction, and never lets this action change asset status. The generated
work order must preserve an observed symptom rather than promote a retrieved
possibility into a confirmed diagnosis.

## Source Layout

```text
aidataapps/rag/
  config/models.json          # pinned model registry and benchmark matrix
  data/                       # fictional source corpus, assets, fixed eval cases
  db/schema.sql               # SQL Server 2025 tables and VECTOR(1024)
  scripts/env-init.sh         # host-to-working-stack initializer
  scripts/model-server.ts     # one-at-a-time vLLM profile manager
  scripts/setup-database.ts   # create, embed, seed, verify
  scripts/benchmark.ts        # artifact-writing API benchmark
  src/agent.ts                # retrieval, structured plan, action gate
  src/inference.ts            # OpenAI-compatible vLLM client
  src/repository.ts           # exact vector search and SQL transaction
  src/app.ts                  # Fastify HTTP contract
  tests/                      # CPU-only unit and contract tests
```

The teaching path and investigation questions are in [COURSE.md](COURSE.md).
Environment evidence belongs in [VALIDATION.md](VALIDATION.md), not in claims
made from an unrecorded terminal session.
