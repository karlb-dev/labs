# Lab 2 Specification Addendum — ModelPrint

## Review, binding corrections, and additions for the implementation and research agent

**Status:** Governing addendum to `aidataapps_modelprint_lab_2_spec.md` (the "spec"). Read both documents before writing code. Where this addendum and the spec conflict, **this addendum wins**. Where this addendum is silent, the spec stands unchanged.

**Review basis:** The spec; the `aidataapps-rag` branch of `karlb-dev/labs` as of commit `a11a433` (Lab 1 README, COURSE, VALIDATION, `config/models.json`, `db/schema.sql`, `src/repository.ts`, `src/inference.ts`, `scripts/model-server.ts`); every prompt source file the spec allowlists, inspected row by row; `interpretability/how_to_design_labs.md`; and the current Microsoft Learn pages for `CREATE VECTOR INDEX`, `VECTOR_SEARCH`, `AI_GENERATE_EMBEDDINGS`, and `CREATE EXTERNAL MODEL` (dated 2026-03-18, last updated 2026-08-12).

**Scientific status:** Unchanged from the spec. Nothing here assumes a fingerprint exists. Several additions exist precisely so that a null result is *interpretable* (no signal) rather than *ambiguous* (no signal the chosen method could see).

### How to read this document

Every item carries an ID and a strength:

- **MUST** — binding. The lab is incomplete without it.
- **SHOULD** — do it unless a recorded reason (budget, capability, stop rule) prevents it. Record the reason in `EXPERIMENT_LOG.md`.
- **MAY** — optional; run only after everything above it in the drop order (§P-5) is frozen.

ID prefixes: `C-` corrections to the spec, `B-` prompt-bank, `G-` generation campaign, `R-` representations and baselines, `S-` SQL program, `E-` evaluation, `P-` experiment program and execution plan, `O-` outputs and reports; the added tests are listed in §9. Each item names the spec section it amends.

---

# 0. Review verdict

## 0.1 What is right and must not be weakened

- The thesis ("a vector search retrieves witnesses; it does not prove authorship") and the separation of retrieval from attribution, calibration, abstention, and clustering (§0, §1).
- Prompt-group-level splits, held-out families/sources/carriers, the leakage audit, and the "same prompt is never a neighbor" rule (§8.8, §8.9, §16, §18.2).
- Exact `VECTOR_DISTANCE` as the scientific reference, ANN as a measured engineering optimization, and the refusal to claim ANN use without plan evidence (§15, §16, §25).
- Versioned model identity (repository + revision + template + parser + image + engine args + decoding) (§2.4).
- Immutable row-level artifacts, idempotent jobs, raw/final/reasoning separation, unique text artifacts (§9, §10).
- Result taxonomy, frozen hypotheses with falsifiers, shuffled-label controls, stop rules, and the claim ceiling (§4, §5, §24.4, §32, §37).
- One chat model resident at a time with a persistent embedding service (§7.3).

## 0.2 The ten problems that change outcomes, ranked

1. **The prompt bank, as allowlisted, cannot deliver the experiment the spec describes.** Measured against the repository: the nine allowlisted sources contain roughly 755 distinct semantic items after deduplication, and about 588 of them (all relation probes and all SAE sentences) are completion-style or single-sentence seeds whose natural answers are 1–10 tokens. Only about 170 items naturally produce medium/long prose. The `full` tier's "100,000 unique segment vectors" target is unreachable from this bank, and the `test_family_holdout` suites would contain tens, not hundreds, of groups. See §B-1 for the counts and the fix (a vendored, licensed, human-written external prompt set with topic-matched human responses).
2. **No supervised baseline on any embedding space.** kNN in a general-purpose semantic space is dominated by topic; that is expected and is H1. But a linear probe on the same vectors often recovers model identity that nearest neighbors cannot retrieve. Without a probe the lab can only report `PROMPT_DOMINATED` when the truth may be "decodable but not retrievable." That distinction is the most interesting SQL-relevant result available, because it motivates storing a *learned* projection in SQL and measuring whether retrieval flips from prompt-aligned to model-aligned (§R-1, §R-2, §S-6).
3. **No likelihood ceiling, despite local access to all four models.** Teacher-forced log-likelihood of every output under every model is prefill-only, costs minutes per model, and gives a 4×4 cross-likelihood matrix plus a Bayes-optimal closed-set reference. Text-only attribution should be reported *relative to that ceiling*. The spec leaves this as an "ambitious extension" (§40.4); it is promoted to a core channel with its own data role (§G-7).
4. **The system-role adapter makes the rendered prompts non-identical across models.** Lab 1 folds the system prompt into the user turn for Gemma (`Instructions:\n…\n\nRequest:\n…`). That textual difference is a model-correlated prompt feature and a prompt-copy leak. The primary campaign must be user-turn-only with no system message; the system-role path becomes an ablation (§G-1).
5. **Decoding is not actually identical across models unless every sampling parameter is sent explicitly.** vLLM's default `--generation-config auto` applies each model's `generation_config.json` defaults (for example a vendor `top_k`) whenever a request omits a parameter, and `repetition_penalty` changes greedy outputs. The campaign must send the complete parameter set on every request and capture the server's effective `SamplingParams` for a canary (§G-2).
6. **Template residue and self-naming can be a trivially perfect fingerprint.** Leftover reasoning tags, role markers, or "As Gemma, …" strings would make attribution look like style when it is lexical leakage. The port gate must scan for them, and a name-masked text view must be available (§G-4).
7. **Mutable serving images.** Three profiles use `vllm/vllm-openai:latest`, and `vllm/vllm-openai:muse-glimmer` is a custom tag with no build recipe in the repository. The frozen registry copy must resolve tags to digests at MP-0 and record the Muse image provenance (§G-5).
8. **Thresholds and holdouts are weaker than the discipline elsewhere.** "Macro-F1 lower bound exceeds 0.25" should be a permutation-null test, not a nominal chance value; a single fixed family holdout should be a leave-one-family-out rotation reported as mean and minimum (§E-1, §E-2).
9. **"Human-authored repository prose" is not a clean human control**; most prose in this repository is AI-assisted. Topic-matched human responses from the external prompt sources are a far better control (§E-4).
10. **ANN may simply be unnecessary at this corpus size**, and the taxonomy has no label for that honest outcome. Add `ANN_UNNEEDED_AT_SCALE` and make the sweep report the crossover, if any (§S-9).

## 0.3 What this addendum adds beyond corrections

- A prompt bank v2 with a controlled repository core plus a vendored external shell, a carrier×source allocation matrix, and a human-response control set (§B).
- One generation rotation instead of three, with nested tiers as frozen subsets (§P-1).
- A cross-likelihood channel (MP-4b), an optional self-recognition sideline, and an "in-the-wild decode" cell (§G-7, §G-8, §G-9).
- Supervised probes, a learned 64-dimensional fingerprint space stored and indexed in SQL, a second embedding family, improved prompt-residual representations, a prompt-centered oracle, and a verbosity-only baseline (§R).
- SQL-native segmentation with `AI_GENERATE_CHUNKS`, phrase log-odds and stylometry in T-SQL, vote aggregation and recall@k as queries, a sampled pairwise-geometry cross join, a learned-space versus raw-space ANN comparison, and an optional in-engine embedding parity check (§S).
- Persona-instructed, paraphrase-laundering, and RAG-grounded robustness suites; mixed-source localization promoted from extension to suite; seed-diversity as a reported result (§E).
- A concrete Colab execution plan with time, disk, and checkpoint budgets, and a results pack that defines what "great reports" means for this lab (§P, §O).

---

# 1. Binding corrections to the spec

## C-1 Prompt bank composition (amends §8.2, §8.5, §8.7) — MUST

**Problem.** Measured contents of the allowlisted sources on the `aidataapps-rag` branch:

| Source | Rows | Distinct semantic items | Natural answer length under `direct-v1` | Group key that must stay together |
|---|---:|---:|---|---|
| `relation_probes_lab1.csv` | 105 | 105 (20 overlap with the next file) | micro (1–3 tokens; completion-style "The capital of France is") | `example_id`; also collapse by relation `category` |
| `advanced_relation_geometry.csv` | 244 | 244 across **12 templates** and 131 subjects | micro | `item_id`, with `family`/`swap_group` treated as the leakage unit |
| `certainty_calibration_items.csv` | 80 | 40 answerable/unanswerable topic pairs, 5 families | short–medium | `topic` (the pair) |
| `steering_eval_prompts.csv` | 24 | 24 open-ended prompts | medium | `prompt_id` |
| `sae_feature_corpus.csv` | 259 | 259 single sentences (~11 words) in 10 domains | micro–short if "rewrite"; medium if "expand" | `text_id` |
| `persona_register_pairs.csv` | 256 | **32** distinct `content_question`s × 8 persona traits | short (instructed under 90 words) | `content_question` |
| `sycophancy_pressure_items.csv` | 240 | **40** base questions × 6 pressure conditions | short ("Answer briefly") | `base_id` |
| `belief_revision_dialogues.csv` | 16 | 16 | short | `item_id` (file already has its own `split` column; ignore it and re-split) |
| `g1_prompts_v1.jsonl` | 15 | 15 | micro–short | `prompt_id` |

Total ≈ 755 distinct items; ≈ 588 are micro seeds. Two further problems: the persona file's `prompt_positive`/`prompt_negative` columns *instruct* a style ("Answer as a patient museum guide"), which is a style confound if used as ordinary prompts; and the relation sets are twelve templates with substituted subjects, so a group-level split still leaks template identity unless the leakage unit is the template family.

**Amendment.** Replace §8.2–8.7 with the prompt bank v2 in §B. In summary: keep the repository sources as the **controlled core** (with the group keys above), add a **vendored external shell** of human-written prompts with permissive licenses and human reference responses, assign carriers by source length class, use persona-instructed variants only as a robustness suite, and size tiers so every held-out suite has at least 150 prompt groups.

**Agent action.** Implement `scripts/vendor-external-prompts.py` (one-time, pinned dataset revisions, frozen output committed under `data/external/`), the per-source group keys, and the allocation matrix in §B-4. The runtime never downloads anything.

## C-2 Primary carriers are user-turn-only (amends §8.6, Risk 5) — MUST

**Problem.** Gemma 4's template rejects the system role; Lab 1 folds the system text into the user turn with `Instructions:`/`Request:` labels. Any system prompt therefore produces a model-correlated difference in rendered text and in the content the model can echo.

**Amendment.** The primary campaign sends exactly one message: `{role: "user", content: <carrier rendering>}`. No system message for any model. Carrier text is byte-identical across the four models. The spec's "system-role adapter" becomes the `system-role-v1` ablation: a dev-tier subset run with a minimal system message on the three profiles that accept one, reported separately (§E-9).

**Agent action.** `src/prompt-bank.ts` renders carriers without a system slot; `src/inference.ts` gains a `completeMessages(messages, decode)` method that sends the message list verbatim (the Lab 1 `complete(systemPrompt, userPrompt)` signature is not reused for the campaign). The render hash covers the full message list.

## C-3 Decoding-parameter hygiene (amends §9.1, §9.3) — MUST

**Problem.** `vllm/vllm-openai` images default to `--generation-config auto`, which loads default sampling parameters from the model's `generation_config.json` and applies them when a request omits the field. Vendor configs commonly ship `top_k`, `top_p`, `temperature`, and sometimes `repetition_penalty`. The spec's deterministic cell only pins `temperature`, `top_p`, and `seed`. `repetition_penalty ≠ 1.0` changes greedy outputs; `top_k` changes the natural cells.

**Amendment.** Every generation request sends the complete set: `temperature`, `top_p`, `top_k` (the "disabled" value for the pinned vLLM version), `min_p = 0`, `repetition_penalty = 1.0`, `presence_penalty = 0`, `frequency_penalty = 0`, `seed`, `max_tokens`, `n = 1`, and explicit `stop = []`. The decode-config hash covers all of them. The port gate captures the vLLM container log line showing the effective `SamplingParams(...)` for one canary request per profile and stores it in the profile snapshot. If any field differs from the request, route to `STOP_PORT`.

**Agent action.** Build requests from a single `DecodeConfig` type; add a unit test asserting every field is present; add `docker logs` capture to the canary gate. Record whether the server ran with `--generation-config auto` or `vllm`; prefer adding `--generation-config vllm` to campaign-time engine args so model defaults cannot apply even by accident, and record that choice in the profile hash.

## C-4 One reference tokenizer for length bands and chunk packing (amends §9.2, §11.1) — MUST

**Problem.** `output_token_count` from each model's own tokenizer is not comparable across models, so length bands would be model-dependent.

**Amendment.** Define `reference_token_count` with one frozen tokenizer for all text: the embedding model's tokenizer (`Qwen/Qwen3-Embedding-0.6B` at the pinned revision), because it is already required for sentence packing. Length bands, `first-64-tokens`, `first-128-tokens`, and segmenter token windows all use the reference tokenizer. Keep the model's own `output_token_count` as a separate column. Also store `output_char_count` and `output_word_count`; length-band claims are made in reference tokens, and the cross-likelihood channel (§G-7) normalizes by characters.

## C-5 Template residue and self-naming scan (amends §9.3, §10.4, §8.3) — MUST

**Problem.** Leftover `<think>`/`</think>`, role markers, tool-call fragments, or explicit self-identification ("I am Gemma, a model developed by Google") would be a lexical leak, not a style signal.

**Amendment.** Add to the port gate: a scan of every canary `final_text` for special-token residue (`<|...|>`, `<think>`, `</think>`, `<reasoning>`, `<tool_call>`, role labels at line start) and for any target model, family, or vendor name. Residue in any canary routes to `STOP_PORT` until the parser or template policy is fixed and re-recorded. Add a text view `name-masked-v1` that replaces model, family, and vendor names with `[MODEL]` and `[VENDOR]`, and report the primary attribution results on `name-masked-v1` alongside `raw-final-v1`. If the two differ materially, label the affected representation `TEMPLATE_ARTIFACT_SIGNAL` (new taxonomy entry, §E-8) and make `name-masked-v1` the headline view.

## C-6 Resolve serving images to digests; record Muse image provenance (amends §6.2, §2.4) — MUST

**Problem.** `vllm/vllm-openai:latest` is mutable; `vllm/vllm-openai:muse-glimmer` is a custom tag with no Dockerfile in the repository.

**Amendment.** `scripts/sync-model-registry.ts` copies Lab 1's registry, bumps `schemaVersion` to 2, and resolves every `vllmImage` tag to `repo@sha256:…` via `docker inspect --format '{{index .RepoDigests 0}}'` after a pull. For `muse-glimmer`, record where the image came from (registry pull or local build); if it is a local build, commit the build recipe under `modelprint/docker/` or route Muse to `STOP_PORT` with the reason. The audit command fails if the frozen copy drifts from the recorded digests.

## C-7 Claim thresholds use a permutation null; family holdout is a rotation (amends §19.6, §32.1, §8.8) — MUST

See §E-1 and §E-2 for the replacement definitions.

## C-8 SQL Server 2025 facts the implementation must respect (amends §14.3, §15, §25) — MUST

Confirmed against the current Microsoft Learn pages (fetched during this review):

- In SQL Server 2025 the vector type, `VECTOR_DISTANCE`, `CREATE VECTOR INDEX`, and `VECTOR_SEARCH` are available; index and approximate search require `ALTER DATABASE SCOPED CONFIGURATION SET PREVIEW_FEATURES = ON`.
- The **latest vector-index version (3)** is currently available only in Azure SQL Database and SQL database in Fabric. Local SQL Server 2025 creates the earlier version. The spec's dual-path design (`ann_legacy` / `ann_v3`) is correct; expect `ann_legacy` to be the one that runs.
- Index version is detected with `JSON_VALUE(v.build_parameters, '$.Version')` from `sys.vector_indexes` joined to `sys.indexes`; `>= '3'` is the latest format. The doctor records this exact query's output.
- Earlier-version indexes: post-filter only, the table becomes read-only after index creation (unless the `ALLOW_STALE_VECTOR_INDEX` database scoped configuration is enabled — never in the primary campaign), and `TOP_N` must be supplied inside `VECTOR_SEARCH`. Version-3 indexes: full DML, iterative filtering, optimizer-driven, and approximate queries must use `SELECT TOP (N) WITH APPROXIMATE` without `TOP_N` (using `TOP_N` against a v3 index raises Msg 42274).
- Both versions: the table must have a primary-key clustered index; at least 100 rows with non-NULL vectors are required (Msg 42266 otherwise); `TRUNCATE TABLE` is not allowed on a table with a vector index (drop the index first); indexes cannot be partitioned. `sys.dm_db_vector_indexes` reports maintenance state. The optimizer may choose a kNN scan instead of the index; on version 3 the `FORCE_ANN_ONLY` table hint forces the index (see the `VECTOR_SEARCH` page for syntax — do not invent it).
- Microsoft's guidance explicitly warns that datasets with many duplicate embeddings degrade DiskANN quality; the spec's unique-text-artifact indexing (§11.4) is therefore also an engineering requirement, not only a scientific one.
- `CREATE EXTERNAL MODEL` + `AI_GENERATE_EMBEDDINGS` and `AI_GENERATE_CHUNKS` exist in SQL Server 2025. External model calls require `sp_configure 'external rest endpoint enabled', 1` and an **HTTPS** endpoint; a plain-HTTP vLLM service needs a TLS proxy. See §S-4 and §S-10.

**Agent action.** Encode every bullet above as a doctor check and as an integration test. Keep search tables loaded and frozen before index creation (spec §15.4). Never benchmark ANN without plan evidence (spec §15.3); on the legacy path capture `SET STATISTICS XML ON` plan XML and assert the vector index object appears in it.

## C-9 System of record versus bulk computation (amends §16, §17, §18) — MUST

**Problem.** Building kNN features for every training row (leave-one-group-out neighbors) means tens of thousands of exact SQL queries, each a full scan of a 1024-dimensional table. That is hours of wall clock for no scientific gain.

**Amendment.** SQL Server is the system of record for all stored vectors, all evaluation-suite queries, all application queries, and the ANN benchmark. Bulk training-time neighbor computation (train-vs-train with prompt-group exclusion) MAY be done in NumPy from vectors exported from the same SQL tables, **provided** an equivalence check proves that for a frozen sample of at least 500 queries the NumPy top-20 neighbor lists and distances match the SQL exact query (same ordering rule, same tie-break, distances within 1e-4, since the engine stores float32). The equivalence report is an artifact. Every test-suite prediction that appears in a report must come from SQL exact search (or SQL ANN in the ANN benchmark), not from NumPy.

## C-10 Bulk loading vectors (amends §14, implementation detail) — SHOULD

Load vectors through a staging table with an `NVARCHAR(MAX)` JSON column using the `mssql` driver's bulk API in batches of 2,000–5,000 rows, then `INSERT … SELECT CAST(json AS VECTOR(n))` in a single set-based statement per batch; verify row counts and a sampled `VECTOR_DISTANCE(v, v) = 0` sanity check after each batch. Row-by-row parameterized inserts are acceptable only for the smoke tier.

## C-11 Python access to SQL Server (amends §6.3, §7.1) — SHOULD

Use `pymssql` (pure pip wheels, no ODBC driver installation) for the analysis package, or have the TypeScript side export Parquet/CSV tables that Python consumes. Do not make the Colab run depend on installing `msodbcsql18` inside the nested runtime. Record the chosen path in `environment/python.json`.

## C-12 Residual representation (amends §12.3) — SHOULD

`normalize(E(output) − E(prompt))` is dominated by whichever of the two has larger norm variance and is not a principled content removal. Implement three prompt-aware variants, all fit or computed on training data only, and compare them:

- `residual-diff-v1`: the spec's difference (kept for continuity).
- `residual-reject-v1`: orthogonal rejection, `o − (o·p̂)p̂` with `p̂ = E(prompt)/‖E(prompt)‖`, then L2-normalize.
- `residual-ridge-v1`: regress `E(output)` on `E(prompt)` with ridge on the train split, store the standardized residual.

Also add the **prompt-centered oracle** `prompt-centered-v1`: `E(output) − mean over the four models' outputs for the same prompt variant and decode cell`. It is computable for every row in the campaign (every prompt is answered by all four models) but not for an external query, so it is a diagnostic upper bound on "signal after perfect content removal," used in geometry, clustering, and probe analyses, never in the app.

## C-13 `--max-num-seqs` and batch sensitivity (amends §9.3, §7.3) — SHOULD

Lab 1 starts vLLM with `--max-num-seqs 8` for latency. A campaign is throughput-bound; run chat models with `--max-num-seqs 64` (Muse: start at 32 and raise if memory allows) and record the value in the profile hash. Greedy decoding under continuous batching is not bitwise batch-invariant, so the repeatability gate runs the 20 canaries twice: alone (sequential) and inside a full batch. Report exact-match rate and mean normalized edit distance for both; declare the cell `deterministic` only if the sequential exact-match rate is ≥ 0.95, otherwise label it `near-deterministic` and say so in every figure caption that uses the deterministic cell.

## C-14 Tier execution (amends §8.7, §28, §35) — MUST

Running `dev`, `standard`, and `full` as separate campaigns means loading each 30B model three times. Tiers become **nested frozen subsets of one bank** (`smoke ⊂ dev ⊂ standard ⊂ full` by prompt-group manifest) and generation happens **once per model residency** at the largest tier the budget allows. Analyses at a smaller tier are computed by filtering rows to that tier's manifest. See §P-1.

## C-15 `max_tokens` and truncation policy (amends §9.1) — MUST

Frozen per carrier (reference-tokenizer estimates; final answers only): `direct-v1` 600, `explain-v1` 600, `structured-v1` 800, `code-explain-v1` 1000, `expand-v1` 500, `rag-grounded-v1` 700, `persona-v1` 400. Muse's reasoning tokens count toward its budget; report `reasoning_token_count` separately and report the rate of `finish_reason = length` with an empty final answer per model as a data-quality metric. Rows with `finish_reason = length` are retained, flagged `truncated = 1`, and excluded from headline attribution claims unless a stratified analysis shows no effect.

## C-16 Pair sets (amends §21.2) — SHOULD

Add the missing cell, `Negative-easy` (different models, different prompt groups, unmatched length), so the pair design is a full 2×2 of {same/different model} × {same/different prompt}, plus the cross-domain and length-matched variants the spec already lists. Report all cells; the gate stays on `Negative-hard`.

---

# 2. Prompt bank v2 (replaces spec §8.2–8.7; §8.1, §8.8, §8.9 stand)

Design principle, restated for this lab: the attribution task is a *classification* problem whose statistical power is set by the number of **independent prompt groups per held-out suite**, not by the number of rendered prompts or samples. Twenty to two hundred carefully paired items is the right microscope for an interpretability lab; it is not enough to calibrate an abstaining four-way classifier across eight test suites. The bank therefore has two layers: a **controlled core** from the repository (pairing structure, provenance, deliberately narrow) and a **breadth shell** vendored once from licensed public prompt sets (natural length diversity, realistic topics, human reference answers).

## B-1 Controlled core (repository sources) — MUST

Use the nine allowlisted files with the group keys in §C-1. Additional rules:

- **Relation probes.** Treat `family` (template) as the leakage unit: all items of one template go to one split. Render them only through prose-forcing carriers (`explain-v1`, `structured-v1`); render at most 100 of them through `direct-v1`, as a capped micro-band sample for the `SHORT_TEXT_INSUFFICIENT` analysis. Convert completion-style text to a question in the carrier ("What is the capital of France?") rather than sending a bare completion prefix to chat models; record the conversion rule and hash.
- **SAE corpus sentences.** Use as `expand-v1` seeds (§B-3), not `rewrite-v1`; single-sentence rewrites are micro outputs.
- **Persona pairs.** The prompt bank uses only `content_question` (32 groups). The `prompt_positive` instructed-style renderings feed the `persona-v1` robustness suite (§E-6) and never the training split.
- **Sycophancy items.** The neutral condition's `question` is the core prompt (40 groups). The five pressure conditions are `pressure` variants of the same group and inherit its split; they are analysed as a nuisance stratum ("user-pressure behavior") and as a within-group robustness check.
- **Certainty items.** Both members of each answerable/unanswerable pair share a group.
- **Steering, belief, g1 prompts.** One group per row.
- **Safety exclusions** per spec §8.3 apply; additionally exclude `refusal_elicitation_set.csv` entirely and use it as a negative keyword reference when filtering the external shell.

## B-2 Breadth shell (vendored external sources) — MUST

Vendor once, at pinned Hugging Face dataset revisions, with file SHA-256s, license text, and the filtering script committed under `data/external/`. The lab runtime never downloads.

| Source | License (verify at the pinned revision) | Use | Target groups |
|---|---|---|---|
| `databricks/databricks-dolly-15k` | CC BY-SA 3.0 | prompts from self-contained categories (`open_qa`, `general_qa`, `brainstorming`, `creative_writing`, `classification` without context); **human `response` as a topic-matched human control** | ~550 |
| `OpenAssistant/oasst1` | Apache 2.0 | English root prompter messages; highest-ranked human assistant reply as a human control | ~300 |
| `nvidia/HelpSteer2` | CC BY 4.0 | prompts only (responses are model-generated) — optional, only if more breadth is needed | 0–200 |

Filters (deterministic, logged as counts in `source-manifest.json`):

1. English only (lightweight language identification; record the tool and threshold).
2. 4–300 words per prompt; drop prompts that require attached context, live data, URLs, dates relative to "today," or file uploads.
3. Drop prompts mentioning any target model, family, vendor, or competitor model name, and prompts about the assistant's identity ("who are you," "what model," "who made you").
4. Drop safety-sensitive prompts via a keyword blocklist seeded from `refusal_elicitation_set.csv` plus standard categories; keep the dropped IDs in the manifest so the decision is auditable.
5. Exact and near-duplicate removal (character-shingle Jaccard ≥ 0.8) within and across all sources, including the repository core.
6. Stratified sampling by Dolly category / OASST length decile with a frozen seed.

Human controls: for every vendored prompt, keep the human response (Dolly `response`; OASST top-ranked reply) in `data/ood/human-authored-controls.jsonl` with the same `prompt_group_id`. These are evaluation-only rows (spec §20.1) and are split with their prompt group, so the human control for a `test_id` prompt is tested against a classifier that has seen the four models answer *other* prompts, never this one.

License note: the repository is MIT; vendored CC BY-SA material must carry its own `LICENSE` and attribution file in `data/external/dolly/`, and derived prompt banks that include Dolly text inherit CC BY-SA. Say so in the README.

## B-3 Carriers v2 (replaces §8.6) — MUST

All carriers are a single user message with byte-identical text across models (§C-2). Versioned templates; the render hash covers template + content + conversion rules.

| Carrier | Purpose | Notes |
|---|---|---|
| `direct-v1` | natural answer, no format instruction | the question or request only |
| `explain-v1` | answer + concise explanation, declared word band (80–150 words) | forces prose from micro seeds |
| `structured-v1` | heading, short answer, rationale, caveat | **held-out carrier** for `test_carrier_holdout` |
| `code-explain-v1` | code + compact explanation + one example | coding items only (persona coding questions, Dolly/OASST code prompts) |
| `expand-v1` | "develop this sentence into a short paragraph (60–120 words)" | replaces `rewrite-v1`; SAE sentences and any sentence-like seed |
| `rag-grounded-v1` | question + 2–3 Lab 1 knowledge chunks as context, "answer using only the context" | Lab 1 tie-in suite (§E-7); includes the four Lab 1 benchmark cases; evaluation-only unless budget allows a training share |
| `persona-v1` | the persona file's instructed-style rendering | robustness suite only, never in train |

Carrier text must not contain model names, the word "assistant" as a role label, or any instruction about length that differs by model.

## B-4 Allocation matrix and tier sizes (replaces §8.7) — MUST

Target `full` composition (the agent finalizes exact counts at freeze after the leakage audit and records them):

| Source class | Groups | Carriers per group | Rendered prompts |
|---|---:|---|---:|
| relation probes (templates as leakage unit) | ~330 | `explain` or `structured` (1 each, alternating by hash) + capped `direct` sample of 100 | ~430 |
| SAE sentences | ~259 | `expand` (1) | ~259 |
| certainty, steering, persona content, sycophancy neutral, belief, g1 | ~167 | `direct` + `explain` + (`structured` or `code-explain` where applicable) | ~450 |
| Dolly | ~550 | `direct` + one of {`explain`, `structured`, `code-explain`, `expand`} by category | ~1,100 |
| OASST1 | ~300 | `direct` (+ `structured` on a hashed 50%) | ~450 |
| `rag-grounded-v1` | ~60 Lab 1 questions | 1 | ~60 |
| **Total** | **~1,650 groups** | | **~2,750 → sample down to 2,500** |

Nested tiers (each is a frozen subset manifest of the one above; generation happens once, §C-14):

| Tier | Groups | Rendered prompts | Decode cells | Outputs (×4 models) |
|---|---:|---:|---|---:|
| `smoke` | 32 | 32 | deterministic | 64 (`qwen-smoke` + one target) |
| `dev` | 200 | 320 | deterministic | 1,280 |
| `standard` | ~1,000 | 1,536 | deterministic + 2 natural seeds | 18,432 |
| `full` | ~1,650 | 2,500 | + high-variance seed | 40,000 |

Expected length mix at `full` (reference tokens): ≥ 60% of outputs in `medium`+`long`. If the dev-tier length report shows less than 50% medium+long, rebalance the carrier matrix before freeze (this is a pre-registered, pre-test adjustment and is recorded as such).

Split sizes at `full` (groups): `train` ≥ 900, `calibration` ≥ 150, `test_id` ≥ 150, `test_source_holdout` = all of one external source (OASST1, ≥ 150), `test_family_holdout` = leave-one-family-out rotation (§E-2), `test_carrier_holdout` = `structured-v1` rows from test groups, `test_decode_shift` = high-variance cell of test groups, `test_ood` = human controls + unknown-model controls + transformed controls.

## B-5 Leakage audit additions (amends §8.9) — MUST

Add `template_family_by_split.csv` (relation templates), `external_source_by_split.csv`, `human_control_by_split.csv`, and `persona_suite_isolation.csv` (proves no persona-instructed rendering is in train). Add a near-duplicate check between every vendored prompt and every repository prompt.

---

# 3. Generation campaign amendments (spec §7, §9)

## G-1 Message construction — MUST

See §C-2. The only per-model difference in the request is the served model name and `chat_template_kwargs` from the registry (`reasoning_strength` for Muse, `enable_thinking: false` for Qwen). Record each model's rendered prompt string for the 20 canaries via the vLLM `/tokenize` endpoint (`messages` + `add_generation_prompt: true`, with token strings) and hash it; this is the "render hash" of spec §9.3. Record the SHA-256 of each model's `chat_template` from `tokenizer_config.json` at the pinned revision in the profile snapshot.

## G-2 Decode configs — MUST

See §C-3. Frozen cells (all fields explicit):

| Cell | temperature | top_p | top_k | min_p | rep. penalty | seeds | samples |
|---|---:|---:|---:|---:|---:|---|---:|
| `det` | 0 | 1 | disabled | 0 | 1.0 | 0 | 1 |
| `nat` | 0.7 | 0.9 | disabled | 0 | 1.0 | 0, 1 | 2 |
| `hv` | 1.0 | 0.95 | disabled | 0 | 1.0 | 2 | 1 |
| `wild` (§G-9) | vendor | vendor | vendor | vendor | vendor | 3 | 1 (dev tier only) |

## G-3 Port gate additions — MUST

Beyond spec §9.3: (a) HF access check for all four repositories at the pinned revisions **before** any download (Gemma is gated; a missing `HF_TOKEN` must fail at MP-0, not at hour six); (b) free-disk check ≥ 1.5 × model size before each cell; (c) effective `SamplingParams` capture (§C-3); (d) template-residue and self-naming scan (§C-5); (e) sequential-versus-batched repeatability (§C-13); (f) image digest match (§C-6); (g) `/tokenize` render hashes for the canaries (§G-1).

## G-4 Text views — MUST

Add `name-masked-v1` (§C-5) and `template-residue-stripped-v1` (used only if residue is found and documented). The primary view remains `raw-final-v1`, with the headline reported on both `raw-final-v1` and `name-masked-v1`.

## G-5 Registry freeze — MUST

See §C-6. Add campaign-time engine-argument overrides (`--max-num-seqs`, `--generation-config vllm`) as a declared `campaignArgs` field in the frozen copy, included in `model_profile_hash`.

## G-6 Residency plan — MUST

One residency per target model, in a fixed order chosen so the model with the slowest download is last. During model *k*'s residency, in this order: port gate → `full`-tier generation for all decode cells → cross-likelihood scoring of every output generated so far by models 1..k and by *k* itself (§G-7) → optional self-recognition items that are already answerable (§G-8) → SQL backup and raw-JSONL mirror to Drive → evict weights. After the rotation, a second short rotation reloads models 1..k−1 only to score outputs generated after their residency (three reloads for four models). If the second rotation is dropped for budget, the cross-likelihood matrix is reported as partial with the missing cells marked, not imputed.

## G-7 Cross-likelihood channel (promotes §40.4 to core) — MUST for `det` and `nat` rows; SHOULD for `hv`

**Why.** With all four models served locally, the teacher-forced log-likelihood of any text under any model is a prefill-only request. It answers a question no text-only method can: *is the model signal recoverable at all, given model access?* Text-only attribution is then reported as a fraction of that ceiling, which makes a null meaningful.

**Data roles.** The likelihood channel is a separate evidence channel with its own tables, metrics, and taxonomy labels. It is **never** a feature of the text-only hybrid (§17 B8). A second hybrid, `B12 hybrid+likelihood`, MAY be reported for rows where scoring exists, clearly labeled "requires model access and is unavailable to the app unless the relevant model is resident."

**Procedure.** For each output text *y* and each scoring model *m*:

1. Render the same user-turn-only prompt *x* with model *m*'s chat template (`apply_chat_template` in Python with the pinned tokenizer, or the `/tokenize` endpoint), append *y* as the assistant message.
2. Request `prompt_logprobs` (vLLM extension) with `max_tokens = 1` and `temperature = 0` (use `/v1/completions` on the rendered string, or chat with `continue_final_message`/`add_generation_prompt: false`; choose one, record it, and test it on a fixture).
3. Locate the assistant-span token boundary by tokenizing the prefix and the prefix+*y* separately with *m*'s tokenizer and taking the count difference (BPE merges at the junction are the reason to do it this way); slice the last *N* prompt logprobs.
4. Store `ll_sum_nats`, `ll_tokens_m` (model-*m* token count), `ll_bits_per_char` (= −ll_sum / (ln 2 · chars)), and `ll_bits_per_ref_token` (reference tokenizer, §C-4).
5. Repeat with an **unprompted** rendering (empty user turn or a fixed neutral user turn "Continue."), giving `ll_unprompted_*`, for prompt-unknown analyses.

Per-character normalization is the comparable quantity across models with different tokenizers; per-token values are for within-model use only.

**Stored representation.** `likelihood-profile-v1`: the 4 prompted + 4 unprompted bits-per-char values, standardized on train, stored as `VECTOR(8)` so that the profile can be retrieved and clustered in SQL like any other vector (a deliberately small demonstration that the engine does not care what the dimensions mean).

**Analyses (MP-6b).**

- 4×4 **cross-likelihood matrix**: mean bits/char of model *j*'s outputs under scorer *i*, with bootstrap CIs; report diagonal advantage and asymmetries (which models "explain" which).
- `B11 Bayes-likelihood`: `argmax_m ll_bits_per_char` (equal priors) as the closed-set reference classifier; accuracy, macro-F1, confusion, by length band and suite.
- Calibrated version: multinomial logistic on the 4-vector, temperature-scaled on `calibration`.
- Ceiling ratio: text-only hybrid macro-F1 ÷ likelihood macro-F1 per suite.
- Mixed-source localization (§E-10) using per-segment likelihoods.

**Budget.** 40,000 outputs × ~300 tokens ≈ 12M prefill tokens per scoring model; at typical prefill throughput that is tens of minutes per model including HTTP overhead. Use `--max-num-seqs 64` and batch 16–32 requests in flight.

**Caveats to print in the report.** Muse is scored without its reasoning span, so its self-likelihood is a "final answer without reasoning" likelihood; Qwen is scored with `enable_thinking: false`; likelihood under a model is evidence about *that served profile*, not the family.

## G-8 Self-recognition sideline — MAY

During the second rotation, for a frozen set of 200 `test_id` prompt groups, ask each resident model to choose which of four shuffled candidate outputs (one per model, det cell) it wrote. Record the choice, log-probabilities of the four option letters, and whether the choice matches (self-recognition accuracy), the model it picked when wrong (preference), and any self-preference in a "which answer is best" variant. Cost: ~200 requests per model. This is an `LLM-says` claim tier, kept separate from the attribution taxonomy, and links the lab to the self-recognition literature (see §11).

## G-9 "In-the-wild" decode cell — SHOULD (dev tier)

Each model is usually deployed with its vendor-recommended sampling settings, which are themselves part of the fingerprint users actually see. Add `wild`: the model's `generation_config.json` defaults (recorded verbatim), seed 3, dev-tier prompts only. Report attribution trained on the frozen identical cells and tested on `wild` as a second decode-shift suite. Never mix `wild` rows into train.

## G-10 Seed diversity as a result — SHOULD

For each prompt variant, compute the semantic and style distance between the two `nat` seeds (and between `nat` and `det`) per model. Report "within-model diversity" by model, carrier, and length. It is a cheap, novel-looking result ("which model is most variable under the same temperature?") and a nuisance covariate for the pairwise task.

---

# 4. Representations, baselines, and hypotheses (spec §12, §17, §4, §5)

## R-1 Supervised probes as baselines (amends §17) — MUST

Add, each fit on `train` with grouped cross-validation for regularization strength, calibrated on `calibration`, evaluated once on frozen suites:

- **B9 semantic probe**: L2-regularized multinomial logistic regression on `semantic1024-whole-v1`.
- **B10 style probe**: the same on `style512-v1` (the spec's B7 is scalar-only; B10 uses the full vector).
- **B11 Bayes-likelihood**: §G-7.
- **B0.5 verbosity-only**: classify from reference token count and paragraph count alone. Any representation that does not beat B0.5 on length-matched rows has learned verbosity, not voice.

Report probes next to kNN for the *same* vector space in every table. The contrast "kNN at chance, probe above chance" is the `PROBE_ONLY_SIGNAL` outcome (§E-8) and is the lab's key bridge result: it means the signal exists in the vectors SQL stores, but cosine nearest-neighbor retrieval cannot surface it, which is exactly the case for a learned projection (§R-2).

## R-2 Learned fingerprint space stored in SQL — MUST

`fingerprint64-v1`: a linear map from the concatenation of `semantic1024-whole-v1` and `style512-v1` (1536 → 64), fit on `train` only, chosen by grouped CV from: (a) neighborhood components analysis on a stratified 5k-row subsample (optimizes kNN accuracy directly), (b) a rank-64 bottleneck softmax regression (reduced-rank multinomial logistic), (c) PCA-256 → LDA-3 padded to 64 as a floor. L2-normalize and store as `VECTOR(64)` in `search_fingerprint_train`, with its own exact baseline and DiskANN index. Evaluate with the same kNN, vote, medoid, geometry, and clustering machinery as the raw spaces, on held-out groups. Hypothesis H11 (§R-6). This is the representation the app should use for witness retrieval if it wins; the raw semantic space remains the representation for "show me similar known outputs."

## R-3 Second embedding family — SHOULD (whole-output space at minimum)

Serve `BAAI/bge-large-en-v1.5` (1024-d, MIT) as a second embedding service alongside Qwen3-Embedding (both fit comfortably next to a 30B chat model on a 96 GB device), or `nomic-ai/nomic-embed-text-v1.5` (768-d, Apache 2.0). Embed at least the whole-output text for all rows; segments if time allows. Report every semantic-space metric for both embedders side by side. This is the only way to say anything about Risk 6 (Qwen embedder favoring Qwen text), and it costs minutes.

## R-4 Residual and oracle representations — SHOULD

See §C-12: `residual-diff-v1`, `residual-reject-v1`, `residual-ridge-v1`, and the diagnostic-only `prompt-centered-v1`.

## R-5 Style512 feature guidance — SHOULD

Keep the spec's dimension allocation. Within the 48 + 48 scalar slots, make sure these LLM-specific tells are explicit features (they are cheap and frequently decisive): em-dash and en-dash rate; curly versus straight quotes; Oxford-comma rate; header count and header style; bold-span count and `**Label:**` line starts; bullet glyph choice and numbered-list rate; blank-line-between-bullets rate; code-fence count and language-tag usage; emoji rate; opener class ("Certainly", "Sure", "Great question", direct start); closer class ("Let me know", "I hope this helps", summary sentence, none); contraction rate; parenthetical rate; colon and semicolon rates; sentence-initial "This/It/Here" rate; mean and variance of sentence and paragraph length; type-token ratio at a fixed window; hedging-word rate; first-person and second-person pronoun rates; ellipsis character versus three periods; trailing whitespace and double-space rate. Store the feature schema with names so the field guide (§O-3) can cite them.

## R-6 Hypotheses added (amends §4) — MUST

- **H9 Probe over retrieval.** On held-out groups, the semantic probe (B9) will exceed semantic kNN (B1) by a margin that survives the permutation null. *Falsifier:* the probe is at the null or does not beat kNN.
- **H10 Likelihood ceiling.** B11 will exceed the best text-only hybrid on every suite, and the gap will widen on short outputs and under carrier shift. *Falsifier:* text-only methods match or exceed likelihood attribution.
- **H11 Learned space retrieves models.** kNN in `fingerprint64-v1` will have a higher retrieval win rate (§24.2) and higher held-out macro-F1 than kNN in either raw space, and DiskANN recall@10 in 64-d will be at least as high as in 1024-d at equal corpus size. *Falsifier:* the learned space does not beat the raw style space on held-out families, or its ANN recall is worse.
- **H12 Instructed style beats intrinsic style.** Persona-instructed prompts (§E-6) will reduce text-only attribution more than they reduce likelihood attribution. *Falsifier:* both channels degrade equally, or neither degrades.
- **H13 Laundering asymmetry.** Paraphrase by a non-target model (§E-6) will collapse phrase and style channels toward chance while the likelihood channel retains above-null signal. *Falsifier:* all channels collapse, or phrase/style survive as well as likelihood.

## R-7 Taxonomy additions (amends §5) — MUST

| Taxonomy | Meaning |
|---|---|
| `PROBE_ONLY_SIGNAL` | Model identity is linearly decodable from a vector space but not retrievable by nearest neighbors in that space. |
| `LIKELIHOOD_ONLY_SIGNAL` | Attribution succeeds with model access (B11) but no text-only channel passes its gate. |
| `TEMPLATE_ARTIFACT_SIGNAL` | Attribution depends on literal template residue or self-naming; headline must use the masked view. |
| `ANN_UNNEEDED_AT_SCALE` | Exact search already meets the application latency budget at the largest achieved corpus; ANN preservation is reported but not recommended. |
| `PARTIAL_LIKELIHOOD` | Cross-likelihood matrix is incomplete because the second rotation was dropped. |

---

# 5. SQL program amendments and additions (spec §14–16, §25)

The goal of this section is to make the SQL layer *scientifically load-bearing* in more places than witness retrieval, without ever letting it substitute for the calibrated layer above it.

## S-1 Doctor additions (amends §15) — MUST

Record, in addition to the spec's list: the exact index-version query from §C-8 and its result; `sys.dm_db_vector_indexes` output; whether `FORCE_ANN_ONLY` parses; whether `REGEXP_LIKE`, `REGEXP_COUNT`, `REGEXP_SUBSTR`, `REGEXP_SPLIT_TO_TABLE` are available (documented as requiring database compatibility level 170 — verify, and record the level, which the spec already does); whether `AI_GENERATE_CHUNKS` runs on a 100-row probe; whether `CREATE EXTERNAL MODEL` parses and whether `external rest endpoint enabled` can be set; `VECTORPROPERTY`, `VECTOR_NORM`, and `VECTOR_NORMALIZE` availability; the maximum vector dimension accepted; and the plan XML of one legacy `VECTOR_SEARCH` query on the probe table, with a boolean `index_operator_present`.

## S-2 Vote aggregation as a query (amends §18) — SHOULD

The exact-neighbor CTE in spec §16 should be completed in T-SQL into the weighted vote the app uses, so that a reviewer can run the classifier's retrieval feature by hand:

```sql
WITH nn AS ( /* spec §16 deduped TOP (@k) neighbors */ )
SELECT model_profile_id,
       SUM(EXP(-distance / @tau))
         / SUM(SUM(EXP(-distance / @tau))) OVER () AS vote_share,
       COUNT(*)                                    AS neighbor_count,
       COUNT(DISTINCT prompt_group_id)             AS independent_groups,
       MIN(distance)                               AS nearest_distance
FROM nn
GROUP BY model_profile_id
ORDER BY vote_share DESC, model_profile_id;
```

`@tau` is the frozen temperature from spec §18.1. The TypeScript repository returns this result set as-is; the hybrid's "vote share" feature is this column, and a contract test asserts the SQL and the NumPy training-time computation agree on a fixture (§C-9).

## S-3 Phrase log-odds in T-SQL (amends §13.2) — SHOULD

Store document-level phrase presence in `phrase_occurrences(phrase_id, generation_id)` for training rows and compute the informative-Dirichlet log-odds ("Fightin' Words", Monroe et al. 2008) as a query, so the model-distinctive phrase table is reproducible inside the engine:

```sql
DECLARE @alpha0 FLOAT = 500.0;  -- prior strength, frozen

WITH docs AS (
    SELECT g.model_profile_id, COUNT(*) AS n_docs
    FROM dbo.generations g WHERE g.split = 'train' GROUP BY g.model_profile_id
),
df AS (
    SELECT o.phrase_id, g.model_profile_id, COUNT(DISTINCT o.generation_id) AS df_model
    FROM dbo.phrase_occurrences o
    JOIN dbo.generations g ON g.generation_id = o.generation_id
    WHERE g.split = 'train'
    GROUP BY o.phrase_id, g.model_profile_id
),
tot AS (
    SELECT phrase_id, SUM(df_model) AS df_all FROM df GROUP BY phrase_id
),
n AS (
    SELECT SUM(n_docs) AS n_all FROM docs
),
scored AS (
    SELECT df.phrase_id, df.model_profile_id,
           df.df_model,
           tot.df_all - df.df_model                  AS df_other,
           d.n_docs                                  AS n_model,
           n.n_all - d.n_docs                        AS n_other,
           @alpha0 * tot.df_all / n.n_all            AS alpha_p
    FROM df
    JOIN tot  ON tot.phrase_id = df.phrase_id
    JOIN docs d ON d.model_profile_id = df.model_profile_id
    CROSS JOIN n
)
SELECT phrase_id, model_profile_id, df_model, df_other,
       LOG((df_model + alpha_p) / (n_model - df_model + @alpha0 - alpha_p))
     - LOG((df_other + alpha_p) / (n_other - df_other + @alpha0 - alpha_p)) AS log_odds,
      (LOG((df_model + alpha_p) / (n_model - df_model + @alpha0 - alpha_p))
     - LOG((df_other + alpha_p) / (n_other - df_other + @alpha0 - alpha_p)))
       / SQRT(1.0 / (df_model + alpha_p) + 1.0 / (df_other + alpha_p))   AS z_score
FROM scored;
```

The Python implementation must reproduce this on a fixture to 1e-9; whichever is the system of record, the other is the check.

## S-4 SQL-native segmentation (amends §11.1) — SHOULD

Add a fifth segmenter, `sql-chunks-v1`, implemented entirely in T-SQL with `AI_GENERATE_CHUNKS(source = …, chunk_type = FIXED, chunk_size = …, overlap = …)` over `text_artifacts`, inserting into `output_segments` with the same row contract (confirm the function's output column names and the meaning of `overlap` against the current docs before relying on them; the doctor's 100-row probe is where that is checked). Compare it with the app-side segmenters on segment count distribution, boundary quality (fraction of segments that start mid-sentence), chunk-vote accuracy, and calibration. It is a genuinely SQL-forward result: does engine-side chunking change an attribution conclusion?

## S-5 Stylometry cross-check in T-SQL (amends §12.4) — MAY

Compute a 12–16 feature subset of `style512-v1`'s scalar block in the engine with `REGEXP_COUNT` (RE2 syntax), for example:

```sql
SELECT generation_id,
       LEN(final_text)                                               AS chars,
       REGEXP_COUNT(final_text, N'(?m)^\s*(?:[-*•]|\d+\.)\s')        AS list_items,
       REGEXP_COUNT(final_text, N'(?m)^#{1,6}\s')                    AS headings,
       REGEXP_COUNT(final_text, N'\*\*') / 2                         AS bold_spans,
       REGEXP_COUNT(final_text, N'—')                                AS em_dashes,
       REGEXP_COUNT(final_text, REPLICATE(N'`', 3)) / 2              AS code_fences,
       REGEXP_COUNT(final_text, N'[.!?](?:\s|$)')                    AS sentence_ends,
       REGEXP_COUNT(final_text, N'\b(?:However|Additionally|Overall|Certainly|Sure)\b') AS discourse_markers,
       REGEXP_COUNT(final_text, N'\b(?:I|my|me)\b')                  AS first_person
FROM dbo.text_artifacts;
```

The point is a parity test (SQL values equal the TypeScript feature values on every row) and a demonstration that a defensible part of the stylometric channel can live next to the data. Not a replacement for `style512-v1`.

## S-6 Learned-space retrieval and ANN (new) — MUST

Build `search_fingerprint_train` (`VECTOR(64)`, §R-2) with the same exact baseline, neighbor dedup rules, and DiskANN index as the raw spaces. Report, side by side for `semantic1024`, `style512`, and `fingerprint64`: retrieval win rate (§24.2), same-prompt versus same-model distance contrast, kNN macro-F1 on held-out families, exact latency, ANN recall@k, and ANN latency at every corpus prefix. This table is the lab's central SQL result: the same engine, the same query shape, and a different stored vector produce retrieval that is either prompt-aligned or model-aligned.

## S-7 Recall@k and decision agreement as queries (amends §25.4) — SHOULD

Persist exact and ANN neighbor lists per benchmark run and compute the retrieval metrics in SQL, so the ANN report is reproducible from tables:

```sql
SELECT e.query_id,
       COUNT(a.neighbor_id) * 1.0 / @k AS recall_at_k
FROM dbo.neighbor_results e
LEFT JOIN dbo.neighbor_results a
       ON a.query_id = e.query_id
      AND a.neighbor_id = e.neighbor_id
      AND a.search_run_id = @ann_run AND a.rank <= @k
WHERE e.search_run_id = @exact_run AND e.rank <= @k
GROUP BY e.query_id;
```

Vote agreement and final-decision agreement join `predictions` for the exact and ANN prediction runs on the same query set.

## S-8 Sampled pairwise geometry in SQL (amends §24) — SHOULD

For each vector space, take a frozen deterministic sample of 2,000 training vectors (by hashed `vector_id`, manifest retained) and compute all ~2M pairwise distances with a self-join and `VECTOR_DISTANCE`, labeling each pair with same-model, same-prompt-group, same-family, same-carrier, same-decode, and length difference. Export to `geometry_pairs.parquet` for the controlled pair regression (§24.3). The NumPy version of the same computation must agree on a 10k-pair subsample; report the SQL wall-clock time as an engineering observation.

## S-9 ANN benchmark realism (amends §25.2, §25.5) — MUST

- Prefix sizes: `1,000 / 10,000 / 25,000 / 50,000 / achieved maximum`. Report the achieved maximum per space honestly; do not pad. The union of unique segment vectors across segmenters in the semantic space is the largest legitimate table and may be the only one that approaches 100k.
- Run the sweep in both 1024-d and 64-d spaces.
- Add the `ANN_UNNEEDED_AT_SCALE` disposition: if exact p95 latency at the achieved maximum is under the app's declared budget (set 250 ms for a single identify call's retrieval step), say so and keep exact as the default regardless of ANN recall.
- Report DiskANN build time and index size per prefix from `sys.dm_db_vector_indexes` and `sys.dm_db_partition_stats`.
- Because the legacy index post-filters, benchmark the oversampling multiplier explicitly: `TOP_N ∈ {k, 2k, 5k, 10k}` with the candidate-loss rate and resulting recall.

## S-10 In-engine embedding parity — MAY (after everything else)

Register the Qwen3 embedding service as an external model (`CREATE EXTERNAL MODEL … WITH (LOCATION = 'https://…/v1/embeddings', API_FORMAT = 'OpenAI', MODEL_TYPE = EMBEDDINGS, MODEL = 'Qwen/Qwen3-Embedding-0.6B')`) behind a TLS sidecar (Caddy or nginx with a certificate the SQL container trusts), and verify that `AI_GENERATE_EMBEDDINGS` returns vectors within 1e-4 cosine distance of the app's vectors for 500 texts. The result is one row in the SQL capability report and a worked example in the README; it is not on the scientific path, and the HTTPS requirement is the reason it is optional.

## S-11 Schema notes (amends §14) — SHOULD

- `generations.split`, `text_view_id`, `reference_token_count`, `truncated`, `template_residue_found`, `self_name_found` columns.
- `likelihood_scores(generation_id, scoring_model_profile_id, prompted, ll_sum_nats, tokens_m, chars, bits_per_char, bits_per_ref_token, scored_at, scoring_run_id)` with a unique key on the first three columns.
- `search_fingerprint_train` and `search_likelihood_train` (`VECTOR(64)`, `VECTOR(8)`).
- `self_recognition(prompt_group_id, judge_model_profile_id, chosen_model_profile_id, option_logprobs_json, correct)` if §G-8 runs.
- Every vector search table carries `corpus_manifest_hash` and a `frozen_at` timestamp; an `INSTEAD OF INSERT/UPDATE/DELETE` trigger that raises an error is the cheapest way to enforce immutability after index creation on the legacy path (where the engine already refuses DML) and on the v3 path (where it would otherwise silently succeed).

---

# 6. Evaluation amendments (spec §19–24, §31–32)

## E-1 Permutation null replaces nominal chance (amends §32.1, §37) — MUST

For every headline metric, compute the null distribution by permuting model labels **within prompt group** (spec §24.4) 200 times with the full pipeline frozen (features, classifier fit, calibration, thresholds). A representation earns `CLOSED_SET_SIGNAL` only if its grouped-bootstrap lower bound exceeds the 95th percentile of the permutation null on `test_id` **and** on the leave-one-family-out rotation (§E-2). Report the null's mean and 95th percentile next to every metric; the nominal 0.25 is printed for orientation only. Apply the same rule to clustering (NMI/ARI), pairwise (AUROC), and the likelihood channel.

## E-2 Leave-one-family-out rotation (amends §8.8, §19.6) — MUST

Family holdout is not one fixed split. Define families as: relation probes, SAE expansions, certainty, steering, persona content, sycophancy, belief, g1, Dolly (by category group), OASST1, and RAG-grounded. For each family *F*: fit on train groups not in *F*, calibrate on calibration groups not in *F*, test on *F*'s test groups. Report mean, minimum, and per-family macro-F1 with CIs; the `SELECTIVE_ATTRIBUTION` gate (spec §19.6) applies to the **minimum** across families with at least 100 test rows. `test_source_holdout` (OASST1 entirely out of train) remains a separate, fixed suite because it is a distribution shift in prompt style, not just topic.

## E-3 OOD sources are split at the source level (amends §19.4, §20.1) — MUST

Tune the novelty threshold on a development set of OOD sources and test on disjoint ones. Development: `qwen-smoke` outputs and the transformed-controls suite. Test: `qwen-3.6-27b-pinned`, human controls (§B-2), mixed-source documents, and any frontier-model prose Karl supplies (§10, decision 7). Report unknown-acceptance per source; expect `qwen-smoke` and `qwen-3.6` to be pulled toward `qwen-3.8-27b` and report that "family attraction" explicitly as a result rather than a failure.

## E-4 Human controls (amends §20.1) — MUST

Replace "human-authored repository prose" with the topic-matched human responses from §B-2. Two analyses: (a) unknown-acceptance rate on human text (the spec's OOD metric); (b) a **descriptive** human-versus-model comparison in each vector space and in the style field guide, labeled as descriptive and outside the closed-set claim ceiling (the spec's non-goal "not a human-vs-machine detector" stands; describing geometry is not building a detector).

## E-5 Calibration metric detail (amends §19.2) — SHOULD

Report ECE with 10 equal-mass (adaptive) bins as the gate metric and 15 equal-width bins for the reliability figure; report classwise ECE; report conformal set size distribution and empirical coverage at α = 0.10 on every suite, with the caveat that coverage is only guaranteed for exchangeable (in-distribution) rows.

## E-6 Robustness suites added (amends §20) — MUST / SHOULD as marked

- **Persona-instructed suite (MUST).** The `persona-v1` renderings of the 32 persona content questions × 8 traits, all four models, det cell. Attribution trained on the standard bank; report by trait. Tests H12. Also report whether the *trait* is more decodable than the *model* in each space (NMI with trait versus NMI with model).
- **Paraphrase laundering (SHOULD).** Paraphrase a frozen 400-row test subset (100 per model, medium/long) with `qwen-smoke` at det settings ("Rewrite the following text in your own words, preserving meaning, length, and format"). Attribute the paraphrases; report per-channel degradation. Tests H13. `qwen-smoke` is already an OOD control and never a training class. Also score the paraphrases under all four models (§G-7) to see whether laundering removes likelihood signal.
- **RAG-grounded suite (MUST, evaluation-only at minimum).** All `rag-grounded-v1` rows, including the four Lab 1 benchmark cases. This is the app-shaped payoff: "can you tell which model wrote the answers in your own RAG logs?" Report it as its own row in the claims table.
- **Transformed controls (spec §20.2) stand.** Add `name-masked-v1` and `template-residue-stripped-v1` to the transformation list.

## E-7 Lab 1 tie-in (new) — SHOULD

The `rag-grounded-v1` carrier reuses `aidataapps/rag/data/knowledge.json` chunks read-only (spec §6.1). Do not run the Lab 1 app or benchmark inside this lab; reproduce grounded generation inside ModelPrint's own campaign so the rows have ModelPrint's provenance. The README's demo section should show `/api/identify` on a Lab 1-style grounded answer with the prompt supplied and with it withheld.

## E-8 Taxonomy and claims table (amends §5, §32, §34) — MUST

Use the additions in §R-7. The claims table gets one row per (representation × method × suite) and a second, shorter table per research question with the adjudicated label. Every claim carries an evidence tag in the house style of `interpretability/how_to_design_labs.md` §1.3, adapted for this lab: `RETRIEVE` (neighbor-level observation), `DECODE` (probe or classifier), `LIKELIHOOD` (requires model access), `CAL` (calibrated decision with abstention), `CLUSTER` (label-hidden), `ENG` (latency/recall engineering), `LLM-SAYS` (self-recognition sideline only).

## E-9 System-role ablation (new; from §C-2) — SHOULD

Dev-tier prompts, det cell, a minimal system message ("You are a helpful assistant."), the three profiles that accept the role. Report the exact-match rate and semantic/style distance between system-role and user-only outputs per model, and whether adding the system role changes attribution of those rows. This quantifies how much Lab 1's fold would have contaminated the experiment.

## E-10 Mixed-source localization (promotes §40.6 to a suite) — SHOULD

Construct 300 documents by concatenating 2–3 paragraphs from different models' answers to the *same* prompt (so topic is constant). Evaluate chunk-level attribution (spec B4) and chunk-level likelihood (§G-7) on paragraph boundaries: per-paragraph accuracy, boundary detection F1, and whole-document "multiple sources present" detection rate. Report as its own claims row; it is the most direct demonstration that chunk vectors carry information the whole-output vector averages away.

## E-11 Clustering additions (amends §23) — SHOULD

Add `prompt-centered-v1` (oracle) and `fingerprint64-v1` (supervised projection; label the run "semi-supervised, held-out groups only") to the clustering representations. The expected pattern — raw semantic clusters align with topic, centered and learned spaces align with model — is worth one figure.

---

# 7. Revised experiment program and Colab execution plan (spec §28, §29, §35)

## P-1 Program (replaces §28 ordering where different) — MUST

| Stage | GPU | Deliverables (delta from spec) | Budget (RTX PRO 6000, approximate) |
|---|---|---|---|
| MP-0 Foundation | no | spec list + digest-resolved registry (§C-6), HF access check, disk plan, doctor with §S-1 additions, `pymssql` path | 1–2 h agent time |
| MP-1 Prompt adapters | no | spec list + group keys (§C-1), external vendoring script and frozen files (§B-2), human-control extraction | 1–2 h |
| MP-2 Freeze | no | spec list + carriers v2, allocation matrix, nested tier manifests, permutation-null plan, LOFO rotation plan, thresholds | 1 h |
| MP-3 Smoke and port gates | yes | `qwen-smoke` end-to-end; per-target canary gate with §G-3 additions | 45 min + ~20 min per target model at first load |
| MP-4 Generation (one rotation) | yes | per residency: gate → `full` generation (all cells) → likelihood scoring of everything generated so far → backup → evict | per model: download 5–15 min, load 3–5 min, generation 30–60 min at `--max-num-seqs 64`, scoring 10–30 min; ~1–2 h per model |
| MP-4b Likelihood completion | yes | second rotation (three reloads) for the missing scoring cells; self-recognition if enabled | ~40 min per reloaded model |
| MP-5 Segmentation, embeddings, features | partial | spec list + `sql-chunks-v1`, second embedder, residual variants, oracle, likelihood profiles | 30–90 min |
| MP-6 Exact retrieval and geometry | no GPU | spec list + probes, learned projection fit, SQL geometry sample, equivalence report (§C-9) | 1–2 h |
| MP-6b Likelihood analysis | no | cross-likelihood matrix, B11, ceiling ratios | 20 min |
| MP-7 Hybrid, calibration, OOD | no | spec list + permutation nulls, LOFO rotation, source-level OOD split | 1–2 h (permutations dominate; parallelize) |
| MP-8 Pairs and grouping | no | spec list + 2×2 pair cells, seed diversity | 30 min |
| MP-9 Clustering | no | spec list + oracle and learned spaces | 30 min |
| MP-10 SQL vector index | no GPU | spec list + 64-d space, oversampling sweep, `ANN_UNNEEDED_AT_SCALE` test | 1–2 h (index builds dominate) |
| MP-11 Application | no | spec list + field guide endpoint, masked-view flag, likelihood block when a model is resident | agent time |
| MP-12 Closeout | no | spec list + results pack (§O), `EXPERIMENT_LOG.md`, `VALIDATION.md` in the Lab 1 house format, Drive mirror inventory | 1–2 h |

GPU phase total: roughly 8–12 hours of wall clock, dominated by model downloads and loads, not by generation. If the VM session cannot be held that long, the checkpoint plan in §P-3 makes every stage resumable.

## P-2 Disk and memory plan — MUST

- One chat model resident (≤ 65 GB weights) + two embedding services (≈ 2–3 GB) + SQL Server. Verify free disk ≥ 1.5 × the next model's size before each download; evict the previous model's weights after its backup is confirmed (Lab 1 already does this on Colab).
- `--gpu-memory-utilization 0.8` for the chat model leaves room for the embedders; confirm with `nvidia-smi` in the port gate and record it.
- SQL storage at `full`: 40k whole vectors × 1024 × 4 B ≈ 160 MB per semantic space; ~150k segment vectors ≈ 600 MB; style 80 MB; fingerprint 10 MB. Trivial. Raw JSONL ≈ 100–200 MB. Keep `runs/` and the SQL backups on the Drive mirror.

## P-3 Checkpoints and resume — MUST

- After every model residency: `BACKUP DATABASE ModelPrint` to a `.bak`, copy it and `raw/generations-<profile>.jsonl` to the Drive mirror, and write `runs/<campaign>/RESUME.md` with the exact next command. The `.bak` is the campaign checkpoint; a new VM restores it and continues with the next profile without regenerating anything.
- Generation and scoring scripts run under `nohup`/`tmux`, checkpoint every 200 rows, and refuse to resume on any governing-hash drift (spec §9.5).
- `EXPERIMENT_LOG.md` is append-only and records every stage start/end time, every gate disposition, every SHOULD that was skipped and why, and every pre-registered adjustment (§B-4 rebalance) with its timestamp relative to the freeze tag.

## P-4 Colab-specific notes — SHOULD

- Reuse Lab 1's `colab-host-init.sh`, rootless daemon, CDI, and `compose.colab.yaml` patterns unchanged except for a second embedding service and the campaign engine args.
- Host networking in the nested runtime means fixed ports: SQL 1433, chat 8000, embedding 8001, second embedder 8002. Record them in `.env.example`.
- The agent's interactive shell can time out; background every GPU job and poll artifacts, do not `await` an HTTP campaign in the foreground.
- Pull all three vLLM image variants (digest-pinned) before the first residency so that an image pull never sits in the middle of a rotation.

## P-5 Drop order (replaces §35) — MUST

1. MP-0 foundation, doctor, digest-pinned registry.
2. MP-1/2 bank v2 and freeze (including external vendoring).
3. Single `full`-tier generation rotation with det + nat cells; `hv` cell if per-model generation time stays under 60 min.
4. Exact semantic, style, and fingerprint baselines with probes; permutation nulls; LOFO rotation.
5. Likelihood scoring during residency (same rotation; near-free). Second rotation (MP-4b) is the first thing to drop → `PARTIAL_LIKELIHOOD`.
6. Calibration, OOD with source-level split, human controls.
7. Pairwise and grouping; mixed-source localization.
8. Chunking comparison including `sql-chunks-v1`.
9. ANN sweep in both dimensionalities.
10. Second embedder segments (whole-output is cheap and stays above this line).
11. Persona, paraphrase, system-role, `wild` suites.
12. Clustering extensions, self-recognition, in-engine embedding parity, active prompt discovery.

Never drop: leakage audits, exact ground truth, raw row retention, calibration, permutation nulls, the masked-view comparison.

---

# 8. Results pack — what "great data, results, and reports" means here (amends §33, §34)

## O-1 Headline table — MUST

`reports/HEADLINE.md` (and `tables/headline.csv`): one row per (representation × method × suite) with macro-F1 and grouped-bootstrap CI, permutation-null 95th percentile, accuracy, top-2 accuracy, ECE, selective accuracy at 50% coverage, coverage at 85% accuracy, and the taxonomy label. Representations: `semantic1024` (both embedders), `style512`, `residual-*`, `prompt-centered` (oracle), `fingerprint64`, `likelihood-profile`, hybrid (text-only), hybrid+likelihood. Methods: kNN vote, medoid, probe, hybrid, Bayes-likelihood. Suites: `test_id`, LOFO mean/min, source holdout, carrier holdout, decode shift (`hv` and `wild`), persona, paraphrase, RAG-grounded, masked-view delta.

## O-2 Figures — MUST (each with a source CSV and a ceiling-aware caption)

```text
F01_length_by_model_carrier.png          output length distributions; truncation rate annotated
F02_cross_model_collisions.png           exact-identical outputs by model pair and length band
F03_geometry_contrasts.png               same-prompt/different-model vs different-prompt/same-model distances, per representation
F04_retrieval_win_rate.png               per representation, by length band
F05_attribution_by_suite.png             macro-F1 with CIs and permutation-null band, per representation × method
F06_confusion_grid.png                   confusion matrices, representation × suite
F07_accuracy_by_length_band.png          with B0.5 verbosity-only baseline overlaid
F08_selective_accuracy_coverage.png      hybrid and best single channels; LOFO minimum
F09_reliability.png                      reliability diagrams, before and after temperature scaling
F10_ood_novelty.png                      novelty ROC by OOD source; unknown-acceptance bars
F11_pairwise_roc.png                     all four pair cells
F12_cluster_alignment.png                NMI with model vs with prompt family/carrier/length, per representation (raw, centered, learned)
F13_ann_recall_latency.png               recall@k and latency vs corpus prefix, 1024-d vs 64-d, exact vs ANN; oversampling sweep inset
F14_cross_likelihood_matrix.png          4×4 bits/char heatmap with CIs; self-likelihood advantage
F15_likelihood_vs_text_only.png          per suite, text-only hybrid as a fraction of the likelihood ceiling
F16_field_guide.png                      top 15 distinctive phrases and top 10 style features per model, with z-scores
F17_seed_diversity.png                   within-model diversity by model and carrier
F18_permutation_nulls.png                null distributions with observed metrics marked
F19_mixed_source_localization.png        per-paragraph accuracy and boundary detection
```

## O-3 Field guide — MUST

`reports/MODELPRINT_FIELD_GUIDE.md`: for each model profile, the top distinctive phrases and layout habits with log-odds and z-scores, three short verbatim example openers and closers (from the lab's own generated text only), the style features with the largest standardized mean differences, and the explicit caveat that every item is a frequency statement, not a signature (spec §13.3 rule 6).

## O-4 README results block — MUST

The lab README ends with a "What we found" section of no more than 400 words, generated from `HEADLINE.md` by a script so it cannot drift from the tables: which representation won on held-out families, the text-only-to-likelihood ceiling ratio, the masked-view delta, the ANN disposition, and a one-sentence statement of what the app can legitimately say about a pasted text. Follow it with a "What we cannot say" list of the spec's non-goals.

## O-5 Experiment log and validation record — MUST

`EXPERIMENT_LOG.md` (§P-3) and `VALIDATION.md` in the Lab 1 format (gate table + environment table + per-profile results table with artifact paths and digests).

## O-6 Reproducibility script — MUST

`repro.sh` must: verify manifests and digests, restore the SQL backup from the mirror, recompute every headline metric from `predictions.parquet`, regenerate every figure from its CSV, diff the regenerated `HEADLINE.md` against the committed one, and exit non-zero on any mismatch. The completion tag is created only after `repro.sh` passes on a fresh checkout (spec §38 last item).

---

# 9. Tests added (amends §30) — MUST

- Request builder emits every decode field (§C-3); fixture asserts equality with a recorded `SamplingParams` line.
- Template-residue and self-name scanners on positive and negative fixtures; `name-masked-v1` is idempotent.
- Reference tokenizer band assignment is independent of the generating model.
- Nested tier manifests satisfy `smoke ⊂ dev ⊂ standard ⊂ full`.
- Registry sync fails on tag-only images and on digest drift.
- Relation templates never straddle splits; persona-instructed rows never enter train; human controls share their prompt group's split.
- Likelihood span slicing: the sliced token count equals the assistant span for three tokenizers on a fixture that includes a junction merge.
- Probe, projection, and calibration fits never see test rows (data-role enforcement test with sentinel rows).
- SQL/NumPy neighbor equivalence on a fixture (§C-9); SQL/Python phrase log-odds equality (§S-3); SQL/TypeScript style feature parity (§S-5 if implemented).
- `sql-chunks-v1` inserts satisfy the segment contract and cover the text exactly once (plus declared overlap).
- Permutation-null driver is deterministic under a fixed seed.
- ANN benchmark rows without plan evidence are labeled `execution_mode_unknown` and excluded from latency claims.

---

# 10. Decisions for Karl before the freeze

Defaults are chosen so the agent can proceed if no answer is given; edit this section to override.

1. **External prompt sources (Dolly, OASST1) vendored into the repository.** Default: **yes**, with licenses and attribution committed. Alternative: repository-only bank with the §B-4 caveats and much smaller held-out suites.
2. **Single `full`-tier rotation** instead of dev → standard → full campaigns. Default: **yes**.
3. **Cross-likelihood second rotation** (three extra model reloads, ~2 h). Default: **yes**; first item to drop under budget pressure.
4. **Second embedding family** (`bge-large-en-v1.5`). Default: **yes**, whole-output space only.
5. **User-turn-only primary carrier.** Default: **yes**; system-role run as a dev-tier ablation.
6. **`wild` decode cell and persona/paraphrase suites.** Default: **yes** at dev tier; drop order item 11.
7. **Additional OOD prose.** If you want to include long-form text you already have from frontier (non-local) models, place it under `data/ood/frontier-prose.jsonl` with a provenance note; it is evaluation-only and tests whether the abstention layer rejects fluent out-of-family text. Default: **omit**.
8. **Self-recognition sideline.** Default: **yes** if the second rotation runs; otherwise skip.
9. **In-engine embedding parity (HTTPS sidecar).** Default: **skip**; capability report notes the requirement.
10. **Muse image provenance.** If `vllm/vllm-openai:muse-glimmer` is a local build, provide the Dockerfile; otherwise the agent records `STOP_PORT` for Muse and proceeds with three models plus `qwen-3.6-27b-pinned` promoted from OOD control to fourth target under a preregistration amendment. Default: **provide the build recipe**.

---

# 11. Updated paste-line for the coding and research agent

> Read `aidataapps_modelprint_lab_2_spec.md` and then `aidataapps_modelprint_lab_2_spec_addendum.md`; where they conflict, the addendum governs. Create branch `aidataapps-modelprint` from `aidataapps-rag`; treat `aidataapps/rag/` and all `interpretability/` data as read-only. Build under `aidataapps/modelprint/`.
>
> Before any target model loads: resolve serving images to digests, verify Hugging Face access to all four pinned revisions, run the SQL capability doctor with the addendum's additions, vendor the external prompt sources at pinned revisions with licenses, build the prompt bank v2 with the per-source group keys and the carrier allocation matrix, freeze nested tier manifests, splits, carriers (user-turn-only, byte-identical), decode cells (every sampling parameter explicit), thresholds, the permutation-null and leave-one-family-out plans, and the preregistration. Commit the freeze tag.
>
> Generate once per model residency at the `full` tier, run the port gate with the addendum's additions, score cross-likelihoods during residency, back up SQL and raw JSONL to the mirror, then evict. Complete the likelihood matrix in a second rotation if budget allows.
>
> Build semantic (two embedders), style, residual (three variants), prompt-centered oracle, learned 64-dimensional fingerprint, and likelihood-profile vectors; load them into frozen SQL search tables; run exact baselines, probes, permutation nulls, LOFO rotation, calibration, OOD with a source-level split, pairs, grouping, clustering, and the ANN sweep in 1024-d and 64-d. Every headline metric is computed from SQL exact search or SQL ANN; NumPy is allowed only for bulk training-time neighbors with the equivalence report.
>
> Finish only when the results pack in addendum §8 exists, `repro.sh` regenerates `HEADLINE.md` byte-for-byte, `EXPERIMENT_LOG.md` explains every skipped SHOULD, and the claims table labels every representation × suite with the taxonomy (including the addendum's additions). A clean null is a successful result; an ambiguous null is not.

---

# 12. References the agent may consult (method inputs, not evidence)

Verify each at the time of use; do not cite anything you have not opened.

- Microsoft Learn: `CREATE VECTOR INDEX (Transact-SQL)`, `VECTOR_SEARCH (Transact-SQL)`, "Vector search and vector indexes in the SQL Database Engine", `sys.vector_indexes`, `sys.dm_db_vector_indexes`, `AI_GENERATE_EMBEDDINGS`, `AI_GENERATE_CHUNKS`, `CREATE EXTERNAL MODEL`, SQL Server 2025 known issues (vector index section), regular-expression functions (SQL Server 2025).
- vLLM documentation: OpenAI-compatible server extra parameters (`prompt_logprobs`, `top_k`, `min_p`, `repetition_penalty`, `seed`), `/tokenize` and `/detokenize`, `--generation-config`, `--max-num-seqs`, reasoning parsers.
- Monroe, Colaresi, Quinn (2008), "Fightin' Words" — informative-Dirichlet log-odds with z-scores (§S-3).
- Sun et al. (2025), "Idiosyncrasies in Large Language Models" — trained classifiers on LLM outputs identify the source model with high accuracy and the signal is robust to several transformations; the reason B9/B10 probes are mandatory and why `PROBE_ONLY_SIGNAL` exists.
- Panickssery, Bowman, Feng (2024), "LLM Evaluators Recognize and Favor Their Own Generations" — framing for the self-recognition sideline (§G-8).
- Hans et al. (2024), "Spotting LLMs with Binoculars" — cross-model perplexity ratios; motivation for reporting the 4×4 matrix rather than only self-likelihood.
- Pasquini et al. (2024), "LLMmap" — active black-box fingerprinting; relevant only to spec §40.1, not to passive attribution claims.
- OpenTuringBench, "From Text to Source," and LLMDet as listed in spec §43.
- Goldberger et al. (2004), Neighbourhood Components Analysis — the learned-metric option for `fingerprint64-v1`.

---

*End of addendum.*
