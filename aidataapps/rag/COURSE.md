# AI Data Apps — Lab 1: Retrieval, Grounding, and Actions

## Lab Thesis

A useful RAG application is not “chat with a vector database.” It is a chain of
contracts:

- which source revision became which chunk
- which embedding model produced the stored vector
- which query retrieved which evidence, at what distance
- which evidence the generator cited
- which output shape crossed into application code
- which caller authorization allowed an operational mutation
- which run artifact makes the observation reproducible

The initial scaffold makes each boundary visible and small enough to replace.

## Starting Experiment

Use the Northstar Bikes corpus to compare four 27B–32B chat models while
holding the embedding model, vectors, SQL query, top-k, prompts, and cases
fixed. A 4B profile is the plumbing control. The serious matrix is Muse Glimmer
30B, Gemma 4 31B, OLMo 3.1 32B Instruct, and Qwen 3.8 27B.

Record:

1. fixed-case pass rate
2. citation validity and required-evidence recall
3. answer versus action classification
4. JSON/contract failure modes
5. warm latency distribution
6. exact model revision and serving-image digest

Do not read a four-case pass rate as broad model quality. The exercise tests
the application boundary and generates failures worth turning into the next
evaluation set.

## Checkpoints

### 1. Inspect the data boundary

Read `data/knowledge.json`, `db/schema.sql`, and the rows produced by
`npm run db:setup`. Explain why document identity, chunk identity, content hash,
and embedding are all stored separately.

### 2. Establish exact retrieval

Run the fixed queries and inspect `VECTOR_DISTANCE('cosine', ...)` results.
Change top-k and write down when additional context helps, distracts, or hides
a missing chunking decision. Exact search is the reference implementation.

### 3. Audit generation grounding

Inspect the model prompt and JSON union in `src/agent.ts`. Introduce one
deliberately invalid citation in a fake model response and confirm that the
application refuses to present it as grounded evidence.

### 4. Cross the action boundary

Issue the same work-order request with actions disabled and enabled. Trace the
proposal, Zod validation, allowlist, caller opt-in, asset lookup, and SQL
transaction. Explain why setting an asset out of service is not bundled into
this action.

### 5. Run the model matrix

Warm each profile, run at least three repeats, and compare the run artifacts.
Separate cold model startup from request latency. When a model fails, preserve
the raw response and classify the failure before changing prompts.

## Likely Follow-On Labs

- exact versus approximate SQL vector search with a corpus large enough to
  justify `CREATE VECTOR INDEX`
- hybrid lexical/vector retrieval and metadata filters
- chunking and embedding-model ablations with a frozen query set
- reranking and context-budget experiments
- action confirmation, idempotency keys, and audit trails
- prompt-injection tests across trusted and untrusted documents
- multimodal maintenance evidence using Muse, Gemma, and Qwen vision inputs
- offline answer/citation grading and regression gates

The next direction should be chosen from observed failure artifacts, not from
framework fashion.
