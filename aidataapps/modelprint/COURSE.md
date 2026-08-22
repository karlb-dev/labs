# Lab 02 Course Guide — ModelPrint

## Learning objectives

By completing this lab, you will be able to:

1. design a leakage-resistant model-output attribution experiment;
2. preserve reproducible model identity across repository revision, tokenizer, template, image, engine, and decoding settings;
3. use SQL Server 2025 vector types and exact distance search as an inspectable retrieval layer;
4. distinguish semantic similarity, deterministic style, learned fingerprints, and model-access likelihood evidence;
5. calibrate a closed-set classifier, construct conformal candidate sets, and abstain on short or out-of-distribution text;
6. measure whether approximate vector search preserves exact retrieval and application decisions;
7. reconstruct a scientific report from immutable row-level artifacts.

## Lab path

### Part 1 — Inspect the contract

Read `README.md`, `MODELPRINT_PREREGISTRATION.md`, and `MODELPRINT_FREEZE_RECORD.md`. Inspect `config/models.json` and the model-registry snapshot. Identify why “Qwen” is not a sufficiently precise class label.

### Part 2 — Verify the environment

Run `./scripts/env-init.sh`, `npm run doctor`, and `npm run check`. Compare the local SQL Server build's supported vector-index syntax with the exact fallback. The capability artifact is evidence; the container tag alone is not.

### Part 3 — Audit the bank

Run `npm run prompts:build`. Review `data/audits/` and answer:

- Which source is held out entirely?
- Why are relation-template aliases co-located?
- Why are persona instructions evaluation-only?
- Why can a human response share a prompt split without entering target-model training?

### Part 4 — Run a model residency

Start one pinned profile, run its port gate, and generate a resumable tier. Inspect one raw JSONL row beside the corresponding SQL generation, attempt, and text-artifact rows. Confirm that final answer and reasoning are distinct fields.

### Part 5 — Build representations

Build whole-output embeddings with both embedding families, deterministic `style512-v1`, prompt residuals, and the learned `fingerprint64-v1`. Inspect the vector dimensions and manifests in SQL. Explain why the prompt-centered representation is an oracle diagnostic and cannot serve an arbitrary app query.

### Part 6 — Evaluate exact retrieval

Run the exact baselines. Compare same-prompt/different-model distance against same-model/different-prompt distance. Then compare kNN with a linear probe. A probe succeeding when kNN fails means the signal is decodable but not naturally retrievable.

### Part 7 — Calibrate and challenge

Fit on train, calibrate on calibration, and evaluate on untouched suites. Review permutation nulls, leave-one-family-out minimum performance, name-masked results, source holdout, human controls, persona, and RAG-grounded outputs. Follow an abstained example through its novelty score and candidate set.

### Part 8 — Measure ANN

Freeze the exact corpus before building an index. Compare exact and approximate neighbors at multiple `k` and oversampling settings. Verify the captured plan and report recall, latency, vote agreement, and final-decision agreement. If exact already meets the latency budget, `ANN_UNNEEDED_AT_SCALE` is a valid result.

### Part 9 — Use the app

Start the Fastify service and exercise identify, compare, group, explore, and evaluation endpoints. Every response must expose the actual search mode and evidence provenance.

### Part 10 — Reconstruct and export

Run report generation and `repro.sh`. Export the database backup and final BACPAC into the run's `database/` directory. Verify the artifact inventory and digests before treating the run as complete.

## Discussion questions

1. What conclusion is justified if semantic vectors cluster by topic while a learned projection classifies model profile above its permutation null?
2. Why can cross-likelihood be a useful ceiling but not a feature of the default text-only app?
3. How does source-level OOD splitting prevent threshold tuning from becoming evaluation leakage?
4. What evidence would make a model-name token a template artifact rather than a genuine style signal?
5. Which operational assumptions make legacy SQL vector indexes require immutable search tables?
