# GhostType experiment log (append-only)

- 2026-08-23 — GT-0 source intake on Karl's Mac. Branch
  `aidataapps-ghosttype` created from `aidataapps-logwarden` @ `bc589713`
  (Lab 03 Tier-1 closeout, Lab 02 final merged). Spec and addendum vendored
  and hashed; dataset package v2 vendored read-only after §B-1 verification:
  588/588 record hashes match, 265/265 legacy IDs preserved, zero
  missing/duplicate. Isolation, port, database, and run-prefix reservations
  in SOURCE_INTAKE.md. Karl-directed adaptations recorded there: local-first
  execution with a frozen MLX mac target registry (Qwen3.8-27B, Muse+DFlash,
  Gemma 4 26B-A4B MoE, OLMo 3.1 32B), Rosetta SQL per §D-1 option 2,
  sequential mac replay reference with E4B as the batch vehicle, Colab
  lift-and-shift as the second campaign. Addendum §12 defaults adopted;
  ScriptDom pinned to latest 170.x at GT-3.

- 2026-08-23 — GT-0.5 scaffold: repository layout per spec section 74;
  compose with the Lab 03 FTS image pattern (SQL port 1435, container
  aidataapps-ghosttype-sqlserver-1, mac overlay); env/runtime scripts ported
  from Lab 03; package.json command surface mirroring the spec section 76
  contract; config/models.mac.json freezes the mac target registry shape
  (final pins land at the port gate). No science yet; foundation only.

- 2026-08-23 — GT-1 complete. SQL foundation up (migrations 001–004 hash-
  tracked; compat 170; Query Store per inherited settings; PREVIEW_FEATURES
  autocommit). Importer re-verified all 588 record hashes in-process, then
  materialized cases, gold candidates, and 428 split groups. Gates: cursor
  round-trip 588/588 after correcting the harness's offset interpretation
  (package offsets are statement-relative; harness fix, gold untouched);
  insertion integrity 588/588; suffix-duplication trap audit 588/588. Two
  package defects found by the split-exclusivity gate and dispositioned as
  role governance (config/dataset-dispositions.json): gt-vector-indexmeta-g03
  → test_id, gt-ops-forced-g03 → test_template_holdout; effective role math
  test_id 335, calibration 44, template_holdout 26. Import is wipe-and-reload
  idempotent for pause/resume. inprogress_lab4.md added as the live
  checkpoint file per Karl's laptop-shutdown contract.

- 2026-08-23 — GT-3 complete. Parse oracle: dotnet 8 JSON-lines service over
  Microsoft.SqlServer.TransactSql.ScriptDom pinned [170.191.0] (TSql170Parser,
  assembly 17.0.191.0; pin recorded in csproj and in every status row per
  addendum B-4). Semantics decided and frozen:
  (a) ghost-text documents are usually incomplete (582/588 empty suffix), so
      errors are classified by position — error 46029 or at/after the last
      meaningful token = trailing incompleteness (allowed); before it =
      structural (real). Gold gate: zero structural errors in the gold
      recomposition.
  (b) recompose-v1 (src/recompose.ts, campaign identity): when the cursor
      line of the prefix is an open `--` line comment, gold/candidate joins
      with a newline — plain concatenation lexically swallows intent-mode
      completions into the comment (376 false fails before this rule; the
      package's own prompt blocks confirm comment-line cursor placement).
  Results over 588: parser_generation 487 pass / 101 not_applicable
  (package parse_eligible=false rows); parse 479 pass / 8 fail / 101 n_a.
  The 8 fails are adjudicated package defects, logged for dataset v2.1,
  golds untouched: 3 syntax-invalid golds (gt-cur-case-03/-04 simple-CASE
  prefix continued as searched-CASE; gt-edge-proc-05 EXEC argument is a
  TRY_CONVERT expression, not allowed by T-SQL grammar) and 5 cursor-
  placement defects where the whole statement is the prefix so gold lands
  after the terminator (gt-sparse-alias-01..04, gt-edge-cont-05; roles:
  4 train + 2 holdout, 1 train among CASE rows — quality aggregates will
  carry a parse-oracle-fail exclusion flag, decided at scoring).
  Status rows in dataset.case_oracle_status (idempotent rewrite);
  manifest manifests/parse-oracle.json; evidence event PASS_WITH_FINDINGS.
  Plan adjustment: GT-2-lite (catalog contexts) moves ahead of GT-5
  baselines because grammar-aware B2 consumes catalog snapshots.

- 2026-08-23 — GT-2-lite complete. Catalog snapshots: migration 005 adds
  catalog.snapshots/objects/columns/extras; scripts/snapshot-catalog.ts loads
  all 14 package catalog files (raw text authoritative + sha256, tolerant
  structured parse for grammar-aware B2 and binding validators). Wipe-and-
  reload idempotent. Results: 14 snapshots, 520 objects, 2,440 columns,
  2 annotation extras, 0 unparsed lines, 0 gate failures. Parser handles
  bracketed/unicode identifiers, unbracketed space names (Northwind views),
  SYSTEM_VERSIONED + PERIOD/HISTORY_TABLE/CONSTRAINT annotations, synonyms,
  names-only lists, system-object shapes (sys./INFORMATION_SCHEMA/
  queryinsights). Cross-audit: 566/566 case must_reference_objects resolve
  against their snapshot's parsed objects. Fixture DDL execution stays
  deferred (dispositioned at intake); catalog text is what prompts embed,
  so the replay plane is unaffected.

- 2026-08-23 — GT-5 complete. Deterministic baselines B0–B4 over all 588
  cases, profile baseline-deterministic, scored into eval.row_scores +
  metric_results through the shared scorer (src/scoring.ts) that model
  replay will reuse. Frozen rules: normalize-v1 (NFC, trim, collapse
  whitespace; case-SENSITIVE because the dataset ships CS-collation edge
  cases) and recompose-v1; parse gate via the pinned ScriptDom service on
  parse-eligible rows. Results (outcomes over 588): B0/B1 pure abstain
  floors (101 abstain_correct = the expect_empty rows); B2 grammar-aware
  catalog: 4 success / 9 partial / 11 fail, category coverage recorded
  per row (alias-dot, schema-dot, table-source, procedure; no_category 545
  — most rows are intent-mode, out of deterministic scope by design);
  B3 history n-gram: 0 success after self-exclusion fix (initial run's 52
  "successes" were train rows matching their own gold — leakage bug found
  and fixed, honest floor is zero); B4 template retrieval: 47 partial,
  0 exact (identifier variance across near-duplicate families).
  Simplifications logged: B4 identifier-safe adaptation reduced to
  same-catalog exemplar reuse (same-catalog identifiers are valid by
  construction); B1 is a frozen keyword-bigram table. H1 floor established:
  no deterministic arm exceeds 4/487 exact on non-empty rows.

- 2026-08-23 — GT-7 replay runner live. scripts/run-quality.ts: raw-text
  transport replaying the package's frozen chat messages exactly (read from
  the hash-verified vendored JSONL); deterministic reference decode
  (temperature 0, top_p 1, no stop strings — stop strings would strip the
  gold's trailing ';' and destroy normalized_exact; class caps
  cursor_fragment 192 / intent_query 384 output tokens); candidate-extract-v1
  (src/extract.ts: one whitespace strip, suffix-overlap trim recorded,
  fence/prose/mode-leak flagged never repaired, empty = abstain); inline
  scoring through the shared scorer + pinned ScriptDom parse delta; raw
  bodies, candidates, decisions, telemetry, and row_scores persisted per
  request in one transaction; completion.requests status is the resume
  cursor (re-run = skip done, failed re-queued). --max-minutes gives the
  20-minute pause contract. Dev findings on gemma-4-e4b (8-row cpu-dev):
  (1) registry corrected — E4B reasoning arrives in the server-split
  `reasoning` field, not inline; (2) reasoning-channel profiles need token
  headroom beyond the content cap or they die in-thought with empty content
  (finish_reason=length) — added per-profile reasoningAllowanceTokens
  (e4b/26b 1280, muse 1024, qwen//no_think 0, olmo 0) and truncation is now
  scored fail/truncation, never abstention; (3) after fixes: 4/8 exact
  success, 4 genuine misses (two being the known gt-cur-case-03/04 gold
  defects; E4B's case-03 output repeats the same simple-CASE trap the
  parse oracle catches — validator working as intended).

- 2026-08-24 — E4B campaign complete (gemma-4-e4b-mlx, M1-packaged, cpu-dev,
  588/588 terminal, 0 request failures, resumed across none — single pass).
  Headline: normalized_exact 17.5% overall — statistically indistinguishable
  from the B2 floor (exact McNemar p=0.91, provisional-adequate) — so H1 is
  NOT supported for the 4B batch vehicle. The per-class decomposition is the
  real finding: cursor_fragment 35/135 exact (25.9%) with 119/126 offered
  candidates parse-clean; intent_query only 5/352 exact (1.4%) yet 201
  partial, 261/352 grounded, 306/327 parse-clean — the model writes
  plausible well-formed SQL that does not string-match gold on full-query
  intent rows. Abstention discipline is the weakness: 63/101 correct
  abstains; 38 expect_empty rows answered (hallucination-resistance miss),
  33 answerable rows wrongly abstained. Early-sample lesson logged: the
  first 73 rows (alphabetically cursor-heavy) showed 48.7% exact — never
  extrapolate from a prefix of an ordered case list. Claims row written
  (H1 NOT_SUPPORTED, e4b). The four 27–32B target profiles are the actual
  H1 test; E4B remains the dev/batch vehicle.

- 2026-08-24 — GT-11 port gates complete. Results: qwen-3.8-27b PASS,
  gemma-4-26b-a4b PASS, olmo-3.1-32b PASS, gemma-4-e4b PASS,
  muse-glimmer-30b WARN. Findings:
  (1) Qwen: the registry's /no_think prefix does NOT disable thinking on
      mlx_lm.server 0.32.1 — all output died in the reasoning field
      (initial gate FAIL with three empty canaries). Fixed with
      chat_template_kwargs {"enable_thinking": false}: clean content,
      stop finish, deterministic, logprobs OK. Registry updated; the
      runner and gate now send chatTemplateKwargs per profile.
  (2) Gate harness fix: mlx servers list a model only after first load, so
      the listing check now re-runs post-canaries (pre-load false-fail).
  (3) Muse WARN: NON-DETERMINISTIC at temperature 0 with the DFlash
      drafter (two identical canary calls differ) — the addendum A-7
      batch-invariance concern realized as speculation nondeterminism.
      Muse campaign rows are labeled non-deterministic-decode; exact
      repeatability claims are out of scope for this profile. Muse also
      abstained on all three synthetic gate canaries while answering a
      plain probe — conservative under the strict canary contract; watch
      abstention behavior on real package prompts during its campaign.
  Pins (system fingerprints, logprobs support, determinism) recorded in
  control.model_profiles.port_gate_json for all five profiles.

- 2026-08-24 — Karl decision: Muse runs WITH the DFlash speculative-decoding
  drafter for the quality campaign ("it's faster and basically the same
  output, so no point not to use it"). The GT-11 WARN stands as a label
  only: Muse rows are non-deterministic-decode; do not disable the drafter
  to chase exact repeatability.

- 2026-08-24 — Qwen campaign complete (qwen-3.8-27b-mlx, M1-packaged,
  mac-quality, 588/588 terminal, 0 request failures, enable_thinking=false).
  Overall normalized_exact 14.3% — numerically BELOW the B2 floor, but the
  decomposition shows why the headline misleads: Qwen is the strongest
  cursor completer so far — 46/135 exact (34.1%, vs E4B 25.9% and B2's
  floor) with 126/127 offered candidates parse-clean — and on intent rows
  13/352 exact / 231 partial / 288 grounded / 335 parse-clean. The collapse
  is abstention discipline: 25/101 correct abstains; on intent-mode trap
  rows it answered 72 of 78 (vs E4B's 33) — it essentially never returns
  empty when the schema context is insufficient, and B0 collects those 101
  rows for free. Latency p50 ~8s cursor / ~12.6s intent. H1 vs B2 on
  overall exact: not supported (p=0.087, direction NEGATIVE — baselineOnly
  79 > modelOnly 58). The per-class contrast (cursor-only H1) will be
  computed as a secondary suite at scoring close; abstention-conditioned
  metrics are the design lesson feeding the confidence/abstention gate
  (addendum A-3).

- 2026-08-24 — Gemma 26B MoE campaign attempt 1 INVALIDATED and re-run.
  588/588 requests completed but 307 rows were reasoning-channel
  truncations (finish_reason=length with empty content): the 1280-token
  allowance calibrated on E4B is far too small for the 26B MoE, whose
  FINISHED rows used up to ~1950 reasoning tokens (max 6830 chars; means
  2.5-3.2k chars). Those 307 rows measure the harness budget, not the
  model. Disposition: allowance raised to 3584 (≈1.8x observed finished
  max), profile campaign rows wiped (raw responses included — they carry
  no valid quality signal), full re-run launched. Non-truncated attempt-1
  rows showed 25 exact / 153 partial / 44 correct abstains, but no
  attempt-1 numbers will be cited. Harness note: truncation-as-fail
  labeling (GT-7) is what made this visible immediately — an
  abstention-coded harness would have reported 307 phantom abstains.

- 2026-08-24 — Gemma 26B attempt 2 complete (588/588) with a residual
  truncation tail: at the 3584 allowance, 155 rows STILL exhausted the
  budget in reasoning (the attempt-1 cut at 1280 had hidden the true
  need). Valid attempt-2 rows: 34 exact / 228 partial / 53 correct
  abstains; cursor 31/135 exact; intent 10/352 exact, 235/352 grounded,
  256 parse-clean; latency is heavy (avg ~28-36s/row — the thinking tax).
  Per addendum A-6 (replay never enforces deadlines; latency recorded,
  judged post hoc) the 155 truncated rows get one final retry tier at
  8192 reasoning tokens, run while the MoE is resident; rows that
  truncate there are dispositioned REASONING_UNBOUNDED as their terminal
  state. Request rows for the 155 reset (children wiped); registry note
  updated with the tier history 1280 -> 3584 -> 8192.

- 2026-08-24 — Gemma 26B campaign CLOSED, 588/588 terminal. Retry tier
  8192 resolved 52 of the 155 truncated rows to real outcomes (2 exact,
  18 partial, others fails-with-content); 103 rows truncated even at
  8192 -> REASONING_UNBOUNDED terminal disposition; 4 rows exceeded the
  600s HTTP timeout mid-decode -> REASONING_UNBOUNDED_TIMEOUT (terminal
  rows written directly; their requests stay status=failed and must not
  be re-queued). Final: exact 18.2% overall (36 answerable exact),
  cursor 31+2/135, H1 vs B2 p=0.91 — not supported. The distinctive
  finding is the reasoning-cost profile: 107/588 rows (18%) cannot
  complete within 8192 thinking tokens or 600s — for a ghost-text
  product this profile is SLO-incompatible on a fifth of the workload
  regardless of answer quality, and that becomes the headline claim for
  this profile rather than exact-match.

- 2026-08-24 — Gemma 26B thinking-off canary (prompted by Karl's find of
  the documented MLX termination bug + chat-template switch). One call
  with chat_template_kwargs {"enable_thinking": false}: clean content
  ("SELECT OrderId FROM dbo.Orders"), finish=stop, 8 completion tokens,
  no reasoning field. Conclusion: the REASONING_UNBOUNDED result is a
  property of the thinking-ON serving configuration; the switch works in
  this MLX stack when passed (we had only applied it to Qwen). New
  registry profile gemma-4-26b-a4b-nothink-mlx added as a separate
  campaign cell in the registry, DOCUMENTED ONLY: its campaign is
  deferred pending Karl's review (host memory pressure; Karl 2026-08-24).
  Active sequence stays OLMo -> Muse -> done.

- 2026-08-24 — Karl's Muse protocol + product framing (pre-registered
  before the Muse campaign). Muse is expected reasoning-heavy (Lab 03
  bench observation). Protocol: start with an 8192 reasoning allowance;
  checkpoint after ~100 rows. If it is drowning (tons of truncations/
  timeouts at 8K) switch to no-reasoning if the template allows — that
  datum ("this model can't be used in this use case in this mode") is
  more useful than a wall of REASONING_UNBOUNDED rows. If it is mid
  (Gemma-like tail) let reasoning-on complete: reasoning vs non-reasoning
  on the same model is a wanted comparison because completion classes
  price latency differently — cursor continuations ("SELECT * FROM ")
  need near-instant responses (likely all local models are too slow
  there), while intent prompts ("-- summarize sales in Oct") tolerate
  much longer waits and correctness dominates, up to user-abandonment.
  This maps onto addendum A-6 deadline-conditioned quality: latency is
  recorded in replay and judged per class post hoc.

- 2026-08-24 — OLMo campaign complete (olmo-3.1-32b-mlx, 588/588, 0
  request failures, 0 truncations — the direct-answerer serving is
  operationally the cleanest of the four). Quality is the weakest:
  overall exact 3.9% (cursor 16/135 = 11.9%, below E4B; intent 4/352);
  highest parse-error rate of any profile (75 offered candidates
  introduce structural parse errors: 39 cursor + 36 intent); abstention
  discipline near zero — answered 98/101 traps including all 78
  intent-mode traps (3 correct abstains). Latency good: ~6.6s cursor /
  ~12.2s intent. Pattern echoes Lab 03 (OLMo needed guided decoding to
  recover contract behavior); here raw-text transport is the frozen
  primary and OLMo's unconstrained behavior costs it. H1 clearly not
  supported; metrics/claims regenerating.
