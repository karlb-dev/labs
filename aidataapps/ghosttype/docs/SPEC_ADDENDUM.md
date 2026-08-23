# Lab 04 Specification Addendum — GhostType

## Review, binding corrections, scope tiers, and additions for the implementation and research agent

**Status:** Governing addendum to `aidataapps_ghosttype_lab_4_spec.md` (the "spec"). Read the spec completely, then this addendum. Where they conflict, **this addendum wins**. Where this addendum is silent, the spec stands.

**Review basis:** The spec; branch `aidataapps-modelprint` at head `92d3597` (matching the spec's reviewed head) including `MODELPRINT_STATE_OF_RECORD.md`, `VALIDATION.md`, and the full `EXPERIMENT_LOG.md`; branch `aidataapps-logwarden` at head `00ae8ae` (2026-08-23, **newer than the spec's reviewed head `c228339`** — the log now contains Muse, Gemma, OLMo, and Qwen residencies, port-gate dispositions, the batching-sequential control, and the queue-deadlock correction); the Lab 02 and Lab 03 addenda; and current pages for ScriptDom packages (TSql170Parser in 170.x; a 180.59.2 package now exists) and Extended Events / vector-index documentation previously vetted.

**Why this addendum leans on live results:** Labs 02 and 03 are mid-flight and have already produced evidence that settles several of the spec's open design choices. Where a Lab 03 result decides a Lab 04 question, this addendum cites the log entry rather than re-arguing from theory.

### How to read this document

- **MUST** — binding. **SHOULD** — do unless a recorded reason prevents it (log it in `EXPERIMENT_LOG.md`). **MAY** — optional, only after the tier above it is closed.
- ID prefixes: `C-` corrections, `T-` scope tiers, `A-` transport/decoding, `B-` dataset and oracles, `K-` baselines/retrieval, `S-` serving plane, `D-` database/platform, `E-` evaluation/statistics, `P-` execution plan, `O-` outputs, `Q-` tests. Each item names the spec section it amends.

---

# 0. Review verdict

## 0.1 What is right and must not be weakened

- **The two-plane split** (§2): frozen replay for quality, live traces for serving, a quality-invariance subset connecting them, and the rule that a faster scheduler must never look more accurate by completing a different sample. This is the correct generalization of Lab 03's capture-once/replay-many.
- **The reframing as a constrained decision and serving system** (§1) and the enumeration of independent failure modes; exact match kept diagnostic beneath a semantic oracle hierarchy (§50, §92).
- **Strong deterministic baselines as the bar** (§35 B0–B4, §59: "not an intentionally weak straw model"), the router as a primary reported arm, and `DETERMINISTIC_FAST_PATH_DOMINANT` as a welcome outcome.
- **The completion-record contract and split-group definition** (§18–19): prefix/suffix as first-class fields, gold as a *set* with a canonical member, must/must-not-reference objects, oracle eligibility per row, and the largest-linked-group split unit.
- **Capability routing** (§17.3, §20.14): a feature missing on the pinned build is an ineligible case, not a model failure.
- **CPU-first with a hard gate before GPU** (§44–46) and the honest `DEV` tag.
- **Inherited conventions** (§9): every one of the fifteen is now backed by a scar in a predecessor log; keep all of them.
- **Cancellation as a first-class contract** (§14) and cancelled work as retained rows.
- **The claim ceiling** (§7), especially "simulated acceptance is not observed user acceptance."

## 0.2 The twelve problems that change outcomes, ranked

1. **The model output protocol will disqualify a model again.** The spec instructs models to "return JSON with insertion text only" (§36) and leaves guided decoding optional (§37, §43.5). Lab 03's log, *today*: OLMo emitted valid JSON on 18/18 unconstrained port-gate requests but omitted the same three required fields every time → `STOP_PORT`, excluded from the primary campaign; the Tier-2 guided-JSON path then passed 9/9 with 6/6 exact repeats → `GUIDED_DECODING_RECOVERS` (receipts `bf46e77…`, `c1ee96f…`). In LogWarden, contract reliability was itself a research question, so that outcome was scientifically meaningful. In GhostType the JSON envelope is *not* the object of study — the deliverable is insertion text. **The primary transport becomes raw insertion text with a class-specific stop policy; guided-JSON is the secondary transport cell; unconstrained-JSON is dropped entirely** (§A-1). This also removes JSON escaping hazards for exactly the Unicode/quoting cases catalog §17.6 stresses.
2. **Scope, again.** Eleven schemas, ~90 tables, ten engine arms × seven retrieval modes, four campaigns, a .NET service, a load runner, and a UI. Lab 03's tier discipline demonstrably worked (today's log uses "Tier-1 port gates" and "Tier-2 --guided-json" operationally). §T defines the Tier 1 Minimum State of Record and the same closeout-before-Tier-2 rule.
3. **Batch invariance is no longer hypothetical.** Lab 03 classified Gemma `REQUEST_LEVEL_NON_INVARIANT` on the batching-sequential control (29/48 identical normalized outputs on byte-identical inputs; receipts `fd9e052…`, `1fc735a…`), while Qwen passed 6/6 exact repeats. H15 and port gate §43.11 are therefore live issues, and the policy must be pre-declared, not discovered: the quality replay's reference profile is **governed-concurrency replay** (the condition the serving plane actually uses), with a frozen 48-case sequential control per model; `NON_INVARIANT` models are labeled and their replay is scoped "batched profile," never silently re-run sequentially (§A-7).
4. **M3 multi-candidate is impossible as written.** §48 permits reusing "one generation call with N candidates" under the deterministic reference decode — but N greedy candidates are N identical strings. Multi-candidate generation requires sampling. M3/M4 get their own frozen, seeded, sampled decode cell, compared against the deterministic single-candidate arms as a separate cell, and the reranker is fit on train-role candidates from that cell (§A-4).
5. **Deadline semantics are unspecified per plane.** If the replay plane enforces `deadlineMs`, slow models lose rows *and* lose quality — a double penalty and a missing-cell generator. Replay records latency and never enforces deadlines; the serving plane enforces them; deadline-conditioned quality is computed *post hoc* from replay latency against each SLO class (§A-6).
6. **Dataset power and composition are unproven.** 588 records over eight roles leaves thin per-family test cells; the 265 legacy rows carry the §15 hazards (prompt-to-answer shape) unless the rebuild genuinely converted them. Require: a legacy-conversion gate, per-catalog × class × role count publication, the Lab 03-style pre-freeze power simulation with per-comparison power labels (adequately-powered vs exploratory), and a dataset v2.1 path via the vendored generators if the primary contrast is underpowered (§B-2, §B-3, §E-1).
7. **Parser generation is a capability axis the spec ties to the wrong thing.** Catalog §17.3 (vector/AI/JSON syntax) can fail ScriptDom parsing through no model fault if the pinned parser generation predates the syntax. `TSql170Parser` exists (ScriptDom 170.x, SQL Server 2025 syntax) and a 180.x package is now published. Pin the exact ScriptDom package version into campaign identity, canary-parse representative §17.3/§17.4 gold rows at intake, and route rows by `(parser_generation_supports, sql_build_supports)` as two independent eligibility flags (§B-4).
8. **The CPU development profile has an unaddressed platform problem.** ModelPrint's log documents "SQL Server 2025's official x86-64 Linux-only container support boundary … for the Mac continuation," and LogWarden now carries `compose.mac.yaml` and an `aidataapps-logwarden-mac` branch. If the CPU profile runs on Apple Silicon, SQL Server 2025 runs only under x86 emulation (functional; slow; every SQL timing is `DEV`), and the local model server is llama.cpp/Ollama/Foundry Local rather than vLLM. Decide the CPU profile's SQL host and model server explicitly (§D-1).
9. **Deterministic baselines must be grammar-state-aware or H1 is a straw man.** B2 as "prefix matching + recent-object ordering" loses to any LLM on cases real IntelliSense handles trivially (after `FROM` → tables; after `alias.` → that table's columns). B2 must consume ScriptDom token context to pick the completion category before ranking names; an optional Tier-2 "industry baseline" adapter over SqlToolsService completion makes H1 mean "does the LLM beat shipping IntelliSense?" (§K-1, §K-2).
10. **Cancellation and confidence both need a concrete mechanism.** Cancelling decode requires streaming requests with client abort — a non-streaming "cancellation" only saves the caller, not the GPU; and candidate confidence needs a declared source: mean token logprob plus validator features, since chat models emit no calibrated score (§A-3, §S-3).
11. **Paid-for mechanics from Labs 02/03 must be inherited explicitly** or they will be re-learned at GPU prices: legacy DiskANN requires exactly one non-nullable four-byte `INT` clustered key (ModelPrint attempt 3); `PREVIEW_FEATURES` and `CREATE VECTOR INDEX` must run autocommit (attempts 1–2); pooled sessions leak `SERIALIZABLE` isolation into later work (today's 73-deadlock storm in LogWarden — fixed by explicit `READ COMMITTED` inside the RPC); BACPAC import drops `PREVIEW_FEATURES` and compatibility level (ModelPrint's restore wrapper now re-applies both); SQL-native chunking split an emoji into an unpaired UTF-16 surrogate that crashed embedding (deterministic U+FFFD repair policy, provenance retained); the full-text image and population-wait discipline from the Lab 03 addendum; the DriveFS finalization race (mirror only checksum-finalized artifacts under unique names with readback). All become Lab 04 requirements (§D-2…§D-5, §K-4).
12. **The serving-tune campaign is combinatorially unbounded as written.** §55's list × four models × the concurrency ladder is weeks of GPU. Bound it: staged tuning on **two** profiles (the fastest and the best-quality from Tier 1), at most **two** frozen final configurations per tuned profile, ngram/prompt-lookup speculation as the primary speculation method (completion context makes it the natural fit; draft-model speculation only where a same-tokenizer draft exists), and the full ladder only where saturation has not already been established (§S-1, §S-4).

## 0.3 What this addendum adds beyond corrections

- Tier structure with a Minimum State of Record and the branch-always-regenerable rule (§T).
- A complete raw-text completion transport: prompt contract, stop policy, suffix-overlap trimming as a frozen versioned normalization, logprob-based confidence, the sampled M3 cell, guided-JSON secondary, reasoning policy for Muse, replay deadline semantics, and the batch-invariance policy grounded in the Gemma result (§A).
- Dataset intake gates: package-hash verification (the dataset is not in this reviewer's hands — only its claimed counts), legacy conversion audit, composition/power publication, parser-axis routing, execution-oracle mechanics (`SET NOEXEC`, `sp_describe_first_result_set` limits) (§B).
- Grammar-aware B2, the SqlToolsService industry baseline option, Lab 03 FTS reuse, ANN scratch-table key rule, embedding input policy (§K).
- A bounded serving plane with one Tier-1 trace, streaming cancellation mechanics, prefix-cache defaults, SLO freeze procedure, and reuse of Lab 03's epoch/receipt/reconciliation machinery, which today's log shows working at 179k-record scale (§S).
- Platform decisions for the Mac CPU profile and the SQL hygiene inheritance list (§D).
- Power labels, ablation-subset bounds, attrition reporting (§E); budgets and residency plan (§P); trimmed results pack (§O); tests (§Q); decisions with defaults (§12); an updated paste-line (§13).

---

# 1. Binding corrections to the spec

## C-1 Raw-text primary transport (amends §36, §37, §43.4–43.5) — MUST
See §A-1. The Lab 03 OLMo result (`STOP_PORT` unconstrained → `GUIDED_DECODING_RECOVERS` guided, same day, receipts in §0.2.1) is the deciding evidence: an envelope failure must not be able to cost a completion lab one of four models.

## C-2 Tiered scope with an MSR closeout (amends Part XVII, §91) — MUST
See §T. The spec's drop order governs compute; this adds a build order.

## C-3 Batch-invariance policy pre-declared (amends §43.11, §52, H15) — MUST
See §A-7. Grounded in Gemma `REQUEST_LEVEL_NON_INVARIANT` and Qwen 6/6 exact repeats from the Lab 03 log.

## C-4 M3/M4 require a seeded sampled cell (amends §48, §35 M3–M4) — MUST
See §A-4. N greedy candidates are identical; the "reuse one generation call" clause is struck for the deterministic cell.

## C-5 Deadlines are serving-plane only (amends §12, §14, §47, §57) — MUST
See §A-6.

## C-6 Dataset intake, legacy conversion, power, and parser axis (amends Part V, §20) — MUST
See §B-1…§B-4 and §E-1.

## C-7 Grammar-aware B2 (amends §35 B2, H1) — MUST
See §K-1.

## C-8 Confidence source and cancellation mechanics (amends §13, §14, §67) — MUST
See §A-3 and §S-3.

## C-9 Inherited SQL mechanics (amends §23, §27, §31, GT-4, GT-6) — MUST
See §D-2…§D-5 and §K-4. Every item cites the predecessor log entry that paid for it.

## C-10 Serving-tune bounds (amends §55, GT-14) — MUST
See §S-1, §S-4.

## C-11 CPU-profile platform decision (amends §41, GT-9) — MUST
See §D-1.

## C-12 Taxonomy additions (amends §68) — MUST
Add: `TRANSPORT_SENSITIVE` (raw-text vs guided-JSON transports disagree beyond tolerance for one profile), `NON_INVARIANT_BATCHED_PROFILE` (quality scoped to the batched reference per §A-7), `PARSER_INELIGIBLE` (row excluded by parser generation, distinct from SQL-build capability), `UNDERPOWERED_EXPLORATORY` (comparison reported without a licensing claim per §E-1), and `SIM_ACCEPTANCE_ONLY` (already implied by §7; make it a taggable disposition).

## C-13 Context-ablation subset bound (amends §49) — MUST
Ablations in §49 run on a frozen, stratified 96–144-case subset (drawn from `test_id` + `test_suffix_holdout` + `test_sparse_catalog`), for all four profiles, under the M2 arm unless the ablation targets another arm — the same bounding rule that kept Lab 03's controls affordable.

## C-14 One vector-dimension decision before migrations (amends §24, §27) — SHOULD
The embedding inheritance is Qwen3-Embedding-0.6B (1024-d) with BGE-large (1024-d) as the ablation family, per ModelPrint. Freeze `VECTOR(1024)` across all retrieval tables and record the BGE 512-token `truncate_prompt_tokens` right-truncation policy from ModelPrint's log as the long-input rule for the ablation family.

---

# 2. Scope tiers and critical path (new; governs Parts XVII and XXI)

## T-1 Tier 1 — Minimum State of Record (MSR) — MUST

| Area | Tier 1 content |
|---|---|
| Stages | GT-0 … GT-13 as specced, with the reductions below; GT-14/15 reduced to the Tier-1 serving slice (§S-1); GT-16 closeout for Tier 1. |
| Schemas/tables | All eleven schemas may exist, but implement only the tables the Tier-1 path touches. Defer: `serving.batch_events/batch_members/scheduler_samples/cache_events` (Tier 2 — vLLM `/metrics` samples stand in), `telemetry.span_events`, `network_samples`, `tempdb_samples`, `file_io_samples` (Tier 2), columnstore (Tier 3), `reporting.report_sections` (fold into snapshots). |
| Principals | Two (`gt_lab`, `gt_service`) with the negative permission tests that matter: the service principal cannot alter dataset/gold/eval rows, cannot execute arbitrary SQL against `GhostTypeControl`, and fixture execution runs under a third throwaway per-fixture principal. Seven-way splits are Tier 2. |
| Dataset | Full GT-1…GT-3: import, hashes, legacy audit (§B-2), all runtime oracle gates, parser-axis routing (§B-4), split/leakage/retrieval-exclusion audits, per-catalog composition report, power simulation (§E-1). |
| Baselines | B0, B1, B2 (grammar-aware), B3, B4 on all eligible rows. SqlToolsService industry baseline is Tier 2. |
| Retrieval | `none`, `catalog-prefix`, `fulltext` (Lab 03 image) or governed BM25, `vector-exact`, `hybrid-rrf`. `hybrid-reranked` and `accepted-history` are Tier 2. ANN is Tier 2 with plan proof. |
| Arms | M0, M1, M2, R0 for all four profiles at the deterministic reference decode; M3 sampled cell on the §C-13 subset for all profiles; M4 Tier 2. |
| Controls | Shuffled schema descriptions, wrong-fixture schema, empty retrieval, suffix removed, prompt-injection comments/descriptions, shuffled gold for the reranker, batch-order permutation (the 48-case control), label permutation. Remaining §60 controls Tier 2. |
| Serving | The Tier-1 slice in §S-1: one closed-loop mixed-class trace at {1, 4, 16} virtual users per profile on the default frozen config, with cancellation-heavy variant on one profile; open-loop, ladder, affinity/randomized caching, chunked prefill, speculation, priorities → Tier 2. |
| Calibration | Confidence model per §A-3 fit on `calibration`; risk-coverage; abstention gates. Conformal Tier 2. |
| Reports | STATE_OF_RECORD, DATASET_REPORT, COMPLETION_QUALITY_REPORT, SAFETY_REPORT, CLAIMS_TABLE, LIMITATIONS, REPRODUCIBILITY, plus preregistration and freeze record. RETRIEVAL and SERVING reports are Tier-1 sections inside the quality/state reports and become standalone at Tier 2. |
| UI | None in Tier 1 (the spec already says the UI is not the source of truth); the evaluation browser and editor demo are Tier 3. |
| Repro | `repro.sh` row-only and database-restore modes, with the ModelPrint restore wrapper behavior (§D-4). |

**Tier 1 closeout:** `repro.sh --mode rows` passes on a fresh checkout; every planned Tier-1 cell has a terminal disposition; the safety suite is clean; deferrals are logged. Only then Tier 2.

## T-2 Tier 2 — SHOULD
Serving tune and final campaigns per §S-1 bounds; guided-JSON transport cell; M4 reranker; `hybrid-reranked` and `accepted-history`; ANN with plan proof and the INT-key scratch rule; SqlToolsService baseline; remaining controls and robustness suites (§60–61); conformal; seven-way principals; standalone retrieval/serving reports; BACPAC; quantized-profile comparison.

## T-3 Tier 3 — MAY
Editor UI and evaluation browser; native FIM arm; code-specialized control; columnstore; speculative draft-model arm; error-prone acceptance profiles; Appendix D directions.

## T-4 Branch rule — MUST
As in Lab 03: a started Tier 2/3 item is reverted or flagged off so Tier-1 `repro.sh` passes at every commit.

---

# 3. Transport, decoding, and replay semantics (amends §36–§38, §43, §47–§48, §52)

## A-1 Primary transport: raw insertion text with a stop protocol (amends §36) — MUST

The model's output **is** the insertion text. No JSON envelope, no fences, no prose. The prompt contract states: emit only the text to insert at the cursor, stop when the insertion is complete. Mechanics, all frozen per completion class:

- **Stop policy.** `line`: stop sequences `["\n"]` plus a statement terminator rule; `token`: `[" ", "\n", ",", "("]`-class stops chosen at freeze from dev-tier evidence; `block`/`statement`: stop on `;`, `GO`-line, or a frozen sentinel the prompt asks the model to emit after the insertion (e.g. `<|done|>` spelled as a plain string) — pick one at freeze, record it, and strip it as part of extraction. `max_tokens` per class: token 12, line 48, block 192, statement 384 (freeze after dev).
- **Extraction and normalization (versioned `candidate-extract-v1`).** Strip at most: one leading/trailing whitespace run, one trailing sentinel, and a **suffix-overlap trim** — if the candidate's tail equals a prefix of the request suffix (the classic FIM duplication failure), trim it and set `suffix_overlap_trimmed = 1` with the trimmed length. Store the raw text and the extracted candidate separately; metrics report both model-emitted and post-extraction outcomes so trimming is measured, never hidden. Any other transformation is a `MODEL_FORMAT` failure, not a repair.
- **Prose/markdown leakage** (a fence, an English sentence, "Here is") is a `MODEL_FORMAT` failure scored against the model — the port gate's code-only canaries (§43.4) must show near-zero leakage before the campaign, so a model that cannot follow the contract is caught at the gate with evidence, as OLMo was in Lab 03.
- **Multi-candidate**: `candidateCount > 1` is served by the sampled M3 cell (§A-4), never by re-asking in the deterministic cell.
- The response envelope of §13 is unchanged — it is the *service's* output, assembled by the gateway from the extracted candidate, validators, and logprobs.

## A-2 Secondary transport: guided JSON (amends §37, §43.5) — SHOULD (Tier 2)

A `transport-guided-json` cell on the M2 arm, all four profiles, §C-13 subset: the same content contract wrapped in a one-field JSON schema (`{"insertText": string}`) via the pinned image's structured-output mechanism (record whether `response_format` json_schema or `guided_json`; Lab 03 generated the schema from the authoritative Zod type — reuse that code). Score agreement with the raw-text transport; disagreement beyond tolerance → `TRANSPORT_SENSITIVE`. This cell exists because Lab 03 proved guided decoding recovers contract failures with exact repeatability; if raw-text leakage is worse than expected for some profile, guided-JSON is the designated fallback and the freeze record says which transport each profile's headline uses.

## A-3 Confidence source (amends §13, §67) — MUST

Request per-token `logprobs` on every generation. The engine score is mean token logprob of the extracted candidate span (token boundaries located by the §9.10 unicode-safe accounting, never by re-tokenizing the prefix separately). Candidate confidence is the calibration model over: engine score, rank margin (sampled cell), candidate length, completion class, parser/binder/safety statuses, retrieval distance and required-object recall, context-truncation flag — exactly the §67 feature list with the engine score made concrete. Fit on `calibration` only; hash-lock before test (Lab 03's pattern, receipt `19ec484…`).

## A-4 The sampled multi-candidate cell (amends §35 M3–M4, §48) — MUST

`decode-sampled-v1`: temperature 0.7, top-p 0.9, explicit disabled top-k/min-p/penalties, seeds {0,1,2}, `n = 3` where the pinned server honors `n`, otherwise three seeded requests. Run on train + calibration + the §C-13 test subset for all profiles. M3 reranks these candidates with deterministic features; M4 (Tier 2) adds the frozen learned reranker fit on train-role candidates only. M3's top-1-after-rerank is compared with the deterministic M2 top-1 as a paired contrast on the shared subset. The deterministic cell remains the headline for M0/M1/M2/R0.

## A-5 Reasoning policy (amends §37) — MUST

Per completion class, reasoning is set to the minimum the profile supports (Muse `reasoning_strength=low`; Qwen `enable_thinking:false`; others n/a), recorded in the decode hash. Reasoning tokens count toward the latency the serving plane reports — if Muse cannot meet the line/block SLO because of reasoning overhead, that is the finding (`QUALITY_GOOD_LATENCY_MISSED`), not a reason to give it a different budget. `reasoning_text` is retained per §9.11 and never enters `insertText`.

## A-6 Deadlines by plane (amends §12, §14, §47) — MUST

Replay: `deadlineMs` is recorded in the packet but **not enforced** — every generation runs to completion, latency is measured, and deadline-conditioned quality (would this candidate have arrived within class SLO?) is computed post hoc per profile and class. Serving plane: deadlines and cancellation are enforced exactly as §14. This keeps the quality matrix complete and makes "quality if we waited" vs "quality within SLO" two reportable numbers instead of one confounded one.

## A-7 Batch-invariance policy (amends §43.11, §52, H15) — MUST

The quality replay runs at governed concurrency (16 workers against `--max-num-seqs 64`, Lab 02/03 settings) and that **is** the reference profile — it is the condition every serving result uses. Per profile, a frozen 48-case sequential control replays byte-identical requests one at a time; classification: exact-output agreement ≥ 45/48 → `INVARIANT_OBSERVED`; below → `NON_INVARIANT_BATCHED_PROFILE`, the headline is scoped "batched profile," and no sequential re-run replaces the campaign. Grounding: Lab 03 measured Gemma at 29/48 under this exact control while Qwen showed 6/6 exact repeats — expect a split verdict and report it plainly.

## A-8 Prompt contract details (amends §36) — MUST

One user message, byte-identical across profiles except registry `chat_template_kwargs`; no system role (Gemma), operating contract in the user turn (Lab 02/03 rule). Cursor is represented by splitting the document into explicit `PREFIX:`/`SUFFIX:` sections (never an in-band marker inside SQL text, which a model could echo); context sections carry provenance labels; the contract forbids referencing objects not listed in the metadata section and states the abstention convention: emitting nothing (empty output followed by stop) is the abstain signal for raw-text transport, recorded as `MODEL_EMPTY → abstain` and scored by the abstention metrics, not as a transport failure.

## A-9 Streaming in replay (amends §71, §20.2 analogue) — SHOULD

Replay uses non-streaming requests (simpler extraction; TTFT null per request; service-level TTFT from `/metrics`). The serving plane uses streaming with `stream_options.include_usage` and client-side TTFT/first-valid-candidate timing — required anyway for cancellation (§S-3).

---

# 4. Dataset and oracle gates (amends Part V, §20, Appendix C)

## B-1 Package verification at intake (amends §16, GT-1) — MUST

The rebuilt package was reviewed *as described by the spec*, not inspected independently — its claimed counts (588/265/323/6) and validation results are inputs to verify, not facts. GT-1 fails to `STOP_DATA` if: any manifest hash mismatches the shipped files; counts differ from the manifest; any legacy ID is missing or content-drifted from `legacy-preservation.json`; or the static-validation artifacts do not reproduce under the lab's own validator. The dataset directory is imported read-only and hashed into the freeze.

## B-2 Legacy conversion gate (amends §15, §16) — MUST

Every one of the 265 legacy rows must satisfy the *same* record contract as new rows: first-class prefix/suffix (suffix may be empty only when the case genuinely ends at EOF), valid cursor offsets, a gold contract with canonical + accepted set, declared oracle eligibility, and a split group. The dataset report publishes, for legacy vs new separately: completion-class mix, suffix-nonempty rate, oracle-eligibility rates, and per-catalog counts. If legacy rows are systematically prompt-to-answer style (empty suffix, statement-class only), the report must say so and the affected claims (H3 suffix lift especially) are scoped to the rows that actually carry suffixes.

## B-3 Composition and role math (amends §19, §48) — MUST

Publish the role × catalog × class × oracle-eligibility grid before freeze. Verify against the §48 matrix that generation covers **all roles** (train and calibration rows need model candidates for M3/M4/calibration — the spec's "588 cases" wording is confirmed to mean all roles, and the freeze record states it). Expected shape to sanity-check: if train+calibration consume ~40% of groups, each named test role holds only a few dozen groups — which is why §E-1's power labels are mandatory, and why the vendored generators are the sanctioned path to a v2.1 expansion (new dataset version, changelog, preserved priors) if the primary contrast is underpowered.

## B-4 Parser generation as an eligibility axis (amends §17.3, §20.5, §32) — MUST

Pin `Microsoft.SqlServer.TransactSql.ScriptDom` to an exact package version (default: latest 170.x stable; the new 180.59.2 line is a decision item, §12.6) and put it in campaign identity. At intake, canary-parse a representative gold row from every catalog family; rows whose gold the pinned parser cannot parse get `parser_generation_supports = 0` and the `PARSER_INELIGIBLE` disposition — they may still run execution-based oracles if the SQL build supports the syntax (the two flags are independent). The ScriptDom service reports its assembly version and `SqlEngineType`/compatibility selection in every result row.

## B-5 Compile and shape oracle mechanics (amends §20.7–20.8, §38.7, §39) — MUST

- Compile/bind check: `SET NOEXEC ON` batch compile in the fixture database (binds objects without executing), recorded as `compile`. `SET PARSEONLY` is not a substitute (no binding).
- Result shape: `sp_describe_first_result_set` where eligible; its known ineligibilities (temp tables, some dynamic constructs) are declared per row as `shape_oracle = unavailable`, never inferred as failure.
- Execution: per §39; add the Lab 03 rule that fixture principals are per-fixture throwaway users and the row/time caps are enforced in the harness, not trusted to the statement.
- Every oracle result row records fixture snapshot hash and oracle implementation version.

## B-6 Retrieval-exclusion and injection audits (amends §19, §20.13, §60.9–60.10) — MUST

The exclusion audit covers gold text, normalized gold, gold ASTs, expected result rows, *and* schema descriptions authored for test-only fixtures. The prompt-injection stress descriptions are marked `adversarial = 1` in `catalog.descriptions` so the safety metric can compute forbidden-offer rate specifically on them, and so they can never be cited as ordinary evidence in reports.

## B-7 Licensed docs corpus (amends §27) — SHOULD

Vendored SQL feature documentation snippets carry per-snippet source URL, retrieval date, and license (Microsoft Learn content is CC BY 4.0 via its public repos — verify per snippet and attribute in `LICENSES.md`). Any snippet without a verifiable license is dropped, not "probably fine."

## B-8 Unicode input policy for embeddings (amends §24, §27; inherits ModelPrint) — MUST

Adopt ModelPrint's exact policy: embedding inputs replace **unpaired** surrogate code units with U+FFFD, retain valid pairs byte-for-byte, and record every repaired ID in the embedding manifest. GhostType's §17.6 catalog guarantees this path fires; the policy plus provenance is the difference between a gate and a mystery crash at MP-5 prices.

---

# 5. Baselines and retrieval (amends §35, §27, §34)

## K-1 Grammar-aware B2 (amends §35 B2) — MUST

B2 consumes the ScriptDom token/context classification (what category completes here: keyword, table source, column of alias X, function, parameter) before ranking names by prefix match, visibility, and recency. Deterministic tie-break, frozen ordering rules, hashed. B1 remains the pure grammar/keyword arm. The dataset report states, per class, B2's category-detection coverage so H1 comparisons are interpretable.

## K-2 Industry baseline (new) — MAY (Tier 2)

An adapter that submits token/line-class cases to a locally hosted SqlToolsService completion endpoint (MIT; the team's own shipping IntelliSense), pinned to an exact release, with the same fixture connection and the candidate list mapped into the B-arm contract. Reported as `B5-industry` with its own scope note. This turns H1 from "beats a lab lookup" into "beats the shipping product's completion," which is the version of the claim worth having — and worth declining if the adapter's cost threatens Tier 1.

## K-3 Full-text and hybrid (amends §27, §34; inherits Lab 03) — MUST

Reuse the Lab 03 FTS image build, catalog/index DDL, population wait, `CONTAINS` query-builder hardening, and the governed app-side BM25 fallback with identical result shape; reuse the RRF-in-SQL pattern with the frozen `k = 60`. Record `actual_mode` per retrieval run exactly as Lab 03 does.

## K-4 ANN mechanics (amends §27, §34; inherits ModelPrint) — MUST (Tier 2 execution)

When the ANN comparator runs: the scratch/search table uses a single non-nullable four-byte `INT IDENTITY` clustered primary key with source IDs as payload (ModelPrint attempt 3's requirement on the legacy index); `PREVIEW_FEATURES` and `CREATE VECTOR INDEX` execute on an explicit autocommit connection (attempts 1–2); plan capture proves the vector-index operator; exact remains the reference and the equivalence sample re-runs per corpus freeze.

## K-5 Embedding budget (amends §24, §27) — SHOULD

Catalog + exemplar + docs corpora are small (thousands of rows); embed both families (Qwen primary, BGE ablation) for the whole corpora rather than rationing — ModelPrint's throughput evidence (520k segment vectors) shows this is minutes, and it removes a whole class of "which family covered which corpus" bookkeeping.

---

# 6. Serving plane (amends Part XI, GT-14/15)

## S-1 Tiered serving program (amends §53–§55, GT-14/15) — MUST

- **Tier 1 (per profile, default frozen config):** the closed-loop mixed-class trace at 1, 4, and 16 virtual users; the cancellation-heavy variant on one profile; SLO attainment, latency waterfalls, queue depth, cancellation waste, and the quality-invariance subset. Budget: ≈ 30–40 min per profile.
- **Tier 2 (two profiles: fastest + best-quality from Tier 1):** staged tuning per §55's order with at most two frozen final configurations per tuned profile; open-loop arrivals; affinity vs randomized caching controls; chunked prefill; the concurrency ladder continued only until the §90 stop gates; ngram speculation (§S-4). Final traces run only the frozen configurations.
- **Tier 3:** draft-model speculation, priority classes, quantized profiles, 32–64-user cells beyond established saturation.

## S-2 Prefix caching defaults (amends §55.7, H11) — MUST

Record whether the pinned image enables automatic prefix caching by default; toggle it **explicitly** in both directions for the affinity/randomized cells so H11 is a controlled comparison, not a default-setting artifact. The schema-affinity trace shares the metadata/context prefix byte-identically across a user group (the context builder must emit stable section ordering for this to work — add a determinism test).

## S-3 Cancellation requires streaming (amends §14, §71) — MUST

The gateway issues streaming requests to the model server and cancels by aborting the connection; the port gate verifies an aborted request stops consuming decode (via `/metrics` running/waiting counts and the abort finish-reason counters) and leaves no orphan lease or late finalization. Non-streaming cancellation is client-side fiction and is not permitted in the serving plane. Discarded-token accounting = tokens generated on aborted requests, from usage-at-abort where available, else the last streamed count.

## S-4 Speculation method (amends §55.11) — SHOULD (Tier 2)

Primary speculation arm: **ngram / prompt-lookup** speculation (per the pinned image's `--speculative-config`), because completion traffic is exactly the workload where the continuation appears verbatim in the prompt (schema names, suffix, exemplars). Draft-model speculation only where a pinned same-tokenizer draft exists (Qwen family), Tier 3. Record acceptance-length distributions; `SPECULATION_NO_LIFT` is a fine outcome.

## S-5 SLO freeze procedure (amends §57) — MUST

The §57 table is a proposal. The freeze uses dev-tier and non-test GPU canaries to set per-class p95 targets once, records the derivation, and never revises after final traces. Report attainment per class per profile per config; `QUALITY_GOOD_LATENCY_MISSED` is assigned per class, not globally.

## S-6 Telemetry machinery (amends §29, GT-15) — MUST

Reuse Lab 03's epoch/receipt/journal/reconciliation implementation — today's log shows it reconciling 300 journals / 179,688 records / 46,778 spans with zero duplicates and hash-verified backups per residency. Do not reinvent it; port the package, keep the receipts format, and keep the bounded-retention + unique-name Drive finalization rules (ModelPrint's BACPAC race fix).

## S-7 Load-runner choice (amends §74 `load/`) — SHOULD

A custom TypeScript runner over the existing gateway client (deterministic seeds, SQL-journaled arrivals) is preferred over k6 for evidence-integration reasons; if k6 is used anyway, its arrival log must be imported into `serving.arrival_events` and reconciled, not summarized.

---

# 7. Database and platform (amends Part VI, §41–§42, GT-4)

## D-1 CPU-profile platform decision (amends §41, GT-9) — MUST

Decide and record before GT-0 which of these hosts the CPU profile's SQL Server (the 2025 container is x86-64 Linux only — documented in ModelPrint's log for the Mac continuation):

1. **x86-64 Linux machine or VM** — full fidelity; timings still `DEV`.
2. **Apple Silicon via x86 emulation** (Docker/colima with Rosetta) — functional for schema, FTS, vector-exact, oracles; slow; every SQL timing is `DEV`-tagged and excluded from any performance sentence; the doctor records the emulation mode.
3. **Remote SQL** (the Colab VM's container reached over a tunnel) — full fidelity, network latency noted.

Default: option 2 if the development machine is the Mac (a `compose.mac.yaml` in the Lab 03 pattern), option 1 if an x86 box is at hand. The CPU model server on Apple Silicon is llama.cpp-server, Ollama, or Foundry Local (all OpenAI-compatible; pick one, pin it, tag `DEV`); vLLM is not assumed off-GPU. The recorded-stub mode (§41) remains the zero-dependency floor for pure lifecycle testing.

## D-2 Connection and isolation hygiene (inherits Lab 03, today) — MUST

Every stored procedure that runs under pooled connections and depends on an isolation level sets it explicitly inside the procedure (`SET TRANSACTION ISOLATION LEVEL READ COMMITTED` for read paths; explicit hints for queue claims). Grounding: Lab 03's calibration invocation failed on 73 deadlocks in 30 seconds because pooled sessions retained `SERIALIZABLE` from evidence transactions into the queue-availability probe; the fix (explicit isolation inside the RPC) plus bounded seeded 1205 retries on **transaction-bounded queue operations only** is inherited verbatim, including the rule that model inference, validators, and finalization are never inside a retry wrapper.

## D-3 DDL autocommit (inherits ModelPrint) — MUST

`ALTER DATABASE SCOPED CONFIGURATION SET PREVIEW_FEATURES = ON`, `CREATE VECTOR INDEX`, and full-text DDL run on explicit autocommit connections in every driver (mssql/Tedious and pymssql both wrapped ModelPrint's attempts in transactions SQL Server rejects).

## D-4 Restore semantics (inherits ModelPrint) — MUST

The restore/BACPAC-import wrapper re-applies compatibility level 170 and `PREVIEW_FEATURES` and re-validates before declaring success (BACPAC provably does not preserve them); BACPAC export drops vector indexes first and recreates after import; full-text repopulation is awaited. Mirroring publishes only checksum-finalized artifacts under unique content-hash names with size/digest readback before and after promotion.

## D-5 Query Store and reserved words (inherits Lab 03/02) — MUST

Apply the Lab 03 Query Store settings (`QUERY_CAPTURE_MODE = ALL`, `INTERVAL_LENGTH_MINUTES = 1`, `DATA_FLUSH_INTERVAL_SECONDS = 60`, flush before materialization, comment-prefix query classes) to both databases. The migration linter rejects unquoted reserved words as aliases or column names (`view` cost ModelPrint a stopped stage) and runs the schema-drift/reserved-word test the spec already names in §30.

## D-6 Two-database boundary tests (amends §21) — MUST

Negative tests: the service principal cannot create, alter, or write in `GhostTypeControl.dataset/eval`; fixture execution cannot reach `GhostTypeControl` at all; model-authored text is never concatenated into SQL (identifier use goes through the validated quoting layer with tests over the §17.6 adversarial identifiers).

---

# 8. Evaluation and statistics (amends Part XIII)

## E-1 Power labels (amends §63–§66; inherits Lab 03 §E-2) — MUST

Before freeze, run the simulation-based power check from dev-tier rows for every predeclared contrast at the frozen minimum effect. Each contrast is labeled `POWERED` or `UNDERPOWERED_EXPLORATORY` in the preregistration; underpowered contrasts are still computed and reported but cannot license a positive claim. If the primary contrast (best LLM arm vs best eligible baseline on `semantic_success_top1`, `test_id`) is underpowered, the sanctioned remedy is a generator-expanded dataset v2.1 before freeze — not a lowered bar.

## E-2 Pairing and attrition (amends §63, §66) — MUST

All model/arm contrasts are paired on identical packets (the spec says this; keep Lab 03's paired grouped bootstrap + sign-flip implementation). Attrition: every planned cell appears in `eval.row_scores` with `eligible`, `outcome`, and `failure_stage`; end-to-end metrics count failures as failures; complete-case metrics show their denominators; a profile with > 10% `MODEL_FORMAT`+`MODEL_EMPTY` on test rows is flagged in the scorecard regardless of its complete-case quality.

## E-3 Simulator scope (amends §50.5, §51) — MUST

Simulated-acceptance metrics carry the `SIM_ACCEPTANCE_ONLY` tag in every table and figure caption; the simulator version and policy hash appear in the freeze; the error-prone profile is Tier 3 and reported apart.

## E-4 Robustness suite bounds (amends §61) — SHOULD

Each robustness suite runs on the §C-13 subset under M2 unless it targets another arm; sampled-decoding robustness reuses the §A-4 cell rather than adding repetitions.

## E-5 Known-world tests (amends §86) — MUST

Keep all ten; add: a synthetic `NON_INVARIANT` profile (the report must classify it), a suffix-duplication profile (extraction must trim and metrics must show both views), and a deadline-post-hoc fixture (replay quality vs within-SLO quality diverge as designed).

---

# 9. Execution plan (amends Parts XVII–XVIII; inherits Lab 02/03 machinery)

## P-1 Budgets (RTX PRO 6000; approximate) — SHOULD

| Stage | GPU | Wall clock |
|---|---|---|
| GT-0…GT-5 (CPU foundation, dataset, fixtures, oracles, baselines) | no | agent-days of build; runtime hours are small |
| GT-6 retrieval + embeddings (both families, all corpora) | embedding only | < 30 min |
| GT-9 CPU campaign | no | ≈ 1–2 h runtime |
| GT-11 port gate per profile | yes | ≈ 15 min + download (Lab 03 measured: 60 GiB in ~3 min at 300+ MB/s, load ~5 s, engine init ~2 min) |
| GT-12 quality replay per profile: (M0+M1+M2+R0) × 588 ≈ 2,350 requests + M3 sampled subset ≈ 400, at 16 workers | yes | ≈ 45–75 min |
| Tier-1 serving slice per profile | yes | ≈ 30–40 min |
| GT-13 scoring, bootstrap, permutations | no | 1–2 h CPU |
| Tier-2 tune + final (two profiles) | yes | ≈ 2–3 h total |

Whole Tier-1 GPU phase ≈ 6–8 h across four residencies; per-residency checkpoint/backup/mirror per Lab 03's epoch pattern.

## P-2 Residency order and reuse — MUST

Inherit the Lab 03 residency epoch machinery (begin/close epochs, sampler-journal ingestion, reconciliation receipts, pinned backups, bounded retention). Order profiles slowest-download-last; pull digest-pinned images before the first residency; the frozen `--gpu-memory-utilization 0.78` and observed KV capacities from the Lab 03 log are the starting points, re-verified per profile.

## P-3 Resume and refusal — MUST

Jobs/attempts per the spec; resume refuses on campaign/freeze/dataset hash drift; the sequential batch-invariance control and the serving traces are separate run kinds so a resumed quality replay can never absorb their rows.

---

# 10. Results pack (amends Parts XX, §88–§89)

## O-1 Headline scorecard — MUST

`SCORECARD.md` + `tables/scorecard.csv`: one row per (profile × arm × transport), columns: eligible/completed/failed; `semantic_success_top1` with paired Δ over best eligible baseline (CI, permutation p, power label); binding precision/recall; nonexistent- and unauthorized-identifier rates; suffix-duplication rate (model-emitted and post-extraction); `MODEL_FORMAT`/`MODEL_EMPTY` rates; selective accuracy at frozen coverage; within-SLO semantic success per class (post-hoc, §A-6); p50/p95 model latency; invariance classification; taxonomy labels. Baselines B0–B4 (and B5 if run) are rows.

## O-2 Tier-1 figures (each with source table, query hash, ceiling-aware caption) — MUST

```text
F01_semantic_success_by_family_arm.png     with baseline band and power labels
F02_lift_over_best_baseline.png            paired deltas per profile×arm, by class
F03_hallucination_matrix.png               nonexistent/unauthorized identifiers, profile×arm
F04_suffix_ablation.png                    prefix-only vs prefix+suffix paired deltas (rows that carry suffixes)
F05_schema_grounding_effect.png            M0 vs M1 vs M2; shuffled/wrong-schema controls overlaid
F06_retrieval_recall_cost.png              required-object recall vs context tokens vs latency, per mode
F07_risk_coverage.png                      per profile, raw vs calibrated
F08_failure_stage_matrix.png               §62 stages, profile×arm
F09_extraction_effects.png                 model-emitted vs extracted candidate outcomes
F10_latency_waterfall.png                  replay decomposition per profile
F11_slo_attainment.png                     Tier-1 serving slice, per class and concurrency
F12_cancellation_waste.png                 cancellation-heavy trace
F13_invariance_control.png                 sequential vs batched agreement per profile
F14_permutation_nulls.png                  observed vs null for licensed claims
```

Tier 2 adds tuning frontiers, cache-affinity controls, throughput-vs-tail, speculation acceptance, and ANN agreement figures.

## O-3 README results block — MUST

Generated from claims/metric rows: which arm/profile beat the best baseline and where; the suffix and schema-grounding verdicts; hallucination rates with and without permission filtering; the invariance split; Tier-1 SLO attainment; what the completion product can honestly offer today; the non-goals list. State explicitly which comparisons were `UNDERPOWERED_EXPLORATORY`.

---

# 11. Tests added (amends Part XIX) — MUST

- Extraction: sentinel strip, whitespace strip, suffix-overlap trim (positive/negative/Unicode fixtures), idempotence, raw-vs-extracted both persisted.
- Stop-policy tables per class are frozen and hash-checked; the prompt contract is byte-identical across profiles (hash test) and contains no system message.
- Logprob span accounting on fixtures with junction merges across three tokenizers (§9.10).
- Sampled cell: seeds honored; `n` fallback to seeded requests produces distinct candidates; greedy `n>1` is rejected by the request builder.
- Deadline non-enforcement in replay (a slow fixture completes and scores) and post-hoc SLO computation.
- Batch-invariance control harness: byte-identical request replay, agreement computation, classification thresholds.
- Parser-axis routing: a §17.3 gold row that the pinned parser rejects is `PARSER_INELIGIBLE`, still execution-eligible when the build supports it.
- Legacy-conversion audit fixtures (a prompt-to-answer-shaped row fails the gate).
- `SET NOEXEC` compile oracle catches a binding error a parse pass misses; `sp_describe_first_result_set` ineligibility is `unavailable`, not fail.
- Isolation hygiene: a deliberately contaminated pooled session (SERIALIZABLE) is neutralized inside queue/read procedures (Lab 03's verification pattern).
- DDL autocommit paths; restore wrapper re-applies preview/compat and refuses otherwise; mirror finalization readback.
- Streaming abort frees decode (metrics-verified) and cannot late-finalize.
- Adversarial identifiers cannot escape the quoting layer; injection-marked descriptions are excluded from ordinary evidence.

---

# 12. Decisions for Karl before the freeze

Defaults let the agent proceed if unanswered.

1. **Raw-text primary transport; guided-JSON secondary; unconstrained-JSON dropped.** Default: **yes** (the OLMo evidence). Flip to guided-JSON primary if you prefer the production-shaped envelope as the headline; raw text then becomes the ablation.
2. **Tier structure and MSR closeout.** Default: **yes**.
3. **Batched-reference invariance policy (§A-7).** Default: **yes**.
4. **Dataset v2.1 expansion via generators if the primary contrast is underpowered.** Default: **yes, before freeze only**.
5. **ScriptDom pin.** Default: latest **170.x** stable; adopt 180.x only after its status is confirmed (you own this component — your call outranks the default).
6. **CPU-profile SQL host** (§D-1). Default: Mac + x86 emulation if the dev box is the Mac; else an x86 Linux host. CPU model server default: llama.cpp-server; Foundry Local is an equally good pick if you want to exercise it.
7. **SqlToolsService industry baseline (B5).** Default: **Tier 2, attempt it** — it is the H1 worth having; drop without guilt if the adapter fights back.
8. **Serving tune scope: two profiles, ≤ two final configs each.** Default: **yes**.
9. **Speculation: ngram primary, draft-model Tier 3.** Default: **yes**.
10. **SLO table values.** Default: freeze from dev canaries per §S-5; the §57 numbers are the starting proposal.
11. **M4 learned reranker.** Default: **Tier 2**; M3 deterministic rerank is the Tier-1 validated arm.
12. **Editor UI.** Default: **Tier 3** — the demo comes after the state of record, as in Labs 02/03.

---

# 13. Updated paste-line for the coding and research agent

> Read `aidataapps_ghosttype_lab_4_spec.md`, then `aidataapps_ghosttype_lab_4_spec_addendum.md`; the addendum governs on conflict. Create `aidataapps/ghosttype/` on branch `aidataapps-ghosttype`; inspect and reuse — do not modify — the ModelPrint and LogWarden implementations, including the residency-epoch/receipt/reconciliation machinery, the FTS image, the restore wrappers, and the queue/isolation fixes recorded in their experiment logs.
>
> Build Tier 1 first and completely: verified dataset intake with legacy-conversion, composition, power, parser-axis, split, leakage, and retrieval-exclusion gates; fixtures, permission-filtered catalog snapshots, and the pinned ScriptDom service; grammar-aware deterministic baselines on every eligible row; exact vector, full-text (or governed BM25), and hybrid retrieval; the completion gateway with raw-text transport, class stop policies, versioned extraction with suffix-overlap trimming, logprob confidence, streaming cancellation in the serving plane, and deadline-free replay; the CPU single-user campaign on the declared platform; port gates; quality replay of M0/M1/M2/R0 for all four profiles at governed concurrency with the 48-case sequential invariance control and the seeded sampled M3 cell; Tier-1 controls; paired scoring with power labels; the Tier-1 serving slice; the Tier-1 reports; and `repro.sh` in both modes.
>
> Close Tier 1 before any Tier 2 item, and keep the branch in a state where the Tier-1 state of record regenerates at every commit. Deterministic engines are serious competitors; a model that cannot follow the raw-text contract fails at the gate with evidence, not mid-campaign; replay is the quality plane and live traces are the serving plane, and their scores never mix. A clean null, a `NON_INVARIANT` label, and `QUALITY_GOOD_LATENCY_MISSED` are all valid results; an unfinished lab is not.

---

# 14. References to verify at time of use (method inputs, not evidence)

- Predecessor evidence (primary): `aidataapps-modelprint` `EXPERIMENT_LOG.md` (DiskANN INT-key, autocommit DDL, BACPAC preview loss, surrogate repair, BGE truncation policy, mirror finalization); `aidataapps-logwarden` `EXPERIMENT_LOG.md` (OLMo STOP_PORT + guided recovery receipts, Gemma non-invariance receipts, Qwen port pass, isolation/deadlock correction, epoch reconciliation).
- ScriptDom: `Microsoft.SqlServer.TransactSql.ScriptDom` NuGet (170.x with `TSql170Parser`; 180.59.2 exists), the open-source SqlScriptDOM repository.
- SQL Server: `SET NOEXEC`, `sp_describe_first_result_set` and its restrictions, full-text on Linux (Lab 03 image), Query Store options, vector index pages as vetted in the Lab 02 addendum, SQL Server 2025 container platform support (x86-64 Linux).
- vLLM (pinned image, currently 0.27.1 per Lab 03): `logprobs`, `n` and sampling params, structured outputs, `stream_options.include_usage`, abort behavior and `/metrics`, automatic prefix caching, `--speculative-config` ngram method, `--scheduling-policy priority`.
- SqlToolsService releases and license, if B5 runs.
- Foundry Local / llama.cpp-server / Ollama OpenAI-compatibility, if chosen for the CPU profile.

---

*End of addendum.*
