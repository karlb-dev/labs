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
