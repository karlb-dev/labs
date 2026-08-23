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
