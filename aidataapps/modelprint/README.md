# ModelPrint

ModelPrint is Lab 02 in the `aidataapps` series: an empirical TypeScript application for closed-set attribution of text produced by four pinned local LLM serving profiles. It asks a deliberately narrow question: given output from one of the four profiles in the frozen campaign, can SQL-backed evidence identify the profile—or should the system abstain?

The lab is also a test of SQL Server 2025 vector workloads. SQL Server is the system of record for prompts, raw generations, immutable text artifacts, vectors, evidence, predictions, and evaluations. Exact `VECTOR_DISTANCE` search is the scientific reference. Approximate vector search is enabled only when the capability probe and captured query plan prove that a vector index was used.

This is not universal authorship detection, a human-versus-AI detector, or proof that a model wrote arbitrary text. A vector search retrieves comparable witnesses; calibration and evaluation determine whether those witnesses support a scoped claim.

## Frozen target matrix

| Profile | Repository | Revision policy |
|---|---|---|
| `muse-glimmer-30b` | `meta-models/Muse-Glimmer-30B` | exact 40-character revision |
| `gemma-4-31b` | `google/gemma-4-31B-it` | exact 40-character revision |
| `olmo-3.1-32b-instruct` | `allenai/Olmo-3.1-32B-Instruct` | exact 40-character revision |
| `qwen-3.8-27b` | `Qwen/Qwen3.8-27B` | exact 40-character revision |

The exact revisions, container image digests, chat-template hashes, generation configuration, and engine arguments are recorded in `data/manifests/model-registry-snapshot.json`. `qwen-smoke` and `qwen-3.6-27b-pinned` are evaluation controls, never target classes.

## Architecture

The VM runs SQL Server 2025, two persistent vLLM embedding endpoints, and one target chat model at a time. Generation requests contain one user message and every sampling parameter explicitly. The application stores the raw response before deriving final text, reasoning text, normalized views, features, and vectors.

```text
frozen prompts -> one resident target model -> raw JSONL + SQL rows
                                              |
                    Qwen/BGE embeddings + deterministic style features
                                              |
                       exact SQL retrieval -> calibrated decision/evidence
                                              |
                           measured SQL ANN path (when proven by plan)
```

## Quick start

The environment script supports ordinary Docker and the rootless Docker-in-Colab profile used by this VM.

```bash
cd /content/labs/aidataapps/modelprint
./scripts/env-init.sh
npm run run:init -- --campaign full
npm run doctor
npm run campaign:freeze -- --config config/campaigns/full.json
```

Run one model residency at a time:

```bash
npm run model -- start --profile muse-glimmer-30b --replace
npm run port:gate -- --profile muse-glimmer-30b
npm run generate -- --profile muse-glimmer-30b --resume
npm run model -- stop
```

Every substantial command accepts or resolves a run beneath `runs/`. The current run is recorded locally in `.current-run`. Run directories contain environment evidence, copied manifests, raw append-only responses, row tables, metrics, figures, reports, database checkpoints, and a final BACPAC; they are ignored by Git because they can be large.

## Prompt and evaluation controls

- Prompt groups, not individual renderings, are split.
- Equivalent relation templates and near-duplicates cannot cross splits.
- OASST1 is a source-level holdout.
- Carriers are byte-identical across target models and use one user turn.
- Same-prompt neighbors are excluded from every evaluation search.
- Headline results are reported on raw and model-name-masked text.
- Calibration uses only the calibration split; test rows never enter fitting.
- Claim gates use grouped bootstrap intervals, 200 within-group label permutations, and leave-one-family-out rotations.
- SQL exact search produces reported test predictions. NumPy may accelerate training-only neighbor construction only after an equivalence test.

## Repository guide

- `config/` — frozen model, campaign, carrier, threshold, and evaluation contracts.
- `data/` — vendored licensed prompt shell, human controls, prompt bank, audits, and manifests.
- `db/` — idempotent migrations and capability-dependent vector-index definitions.
- `scripts/` — environment, freeze, serving, generation, feature, evaluation, export, and reproduction entry points.
- `src/` — TypeScript contracts, application logic, SQL repository, API, and deterministic features.
- `analysis/` — Python fitting, evaluation, and report generation.
- `docs/` — verbatim governing specification and addendum.

## Current scientific status

The campaign is preregistered; results are not asserted in this README until they are reconstructed from retained prediction rows. The final “What we found” block is generated from `reports/HEADLINE.md`, not written by hand.

## What we cannot say

- The app cannot prove authorship or identify arbitrary models outside the frozen set.
- A family label is not interchangeable with a pinned served profile.
- A nearest-neighbor vote is not a calibrated probability.
- A visually separated projection is not evidence of reliable clustering.
- ANN results are not claimed unless the executed query plan names the vector index.
