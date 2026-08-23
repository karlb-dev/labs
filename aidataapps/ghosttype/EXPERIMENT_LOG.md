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
