# GhostType SQL completion dataset v2

This package contains **588** SQL completion records for Lab 04.

- Legacy records retained by stable ID: **265**
- New synthetic expansion records: **323**
- New expansion catalogs: **6**
- Package version: `2.0.0`
- Unified JSONL: `records/ghosttype_sql_completions_v2.jsonl`
- New-only JSONL: `records/ghosttype_synthetic_expansion_v2.jsonl`

The original v1.1.0 package is preserved under `legacy/` with its source SHA-256 in `manifests/legacy-preservation.json`.

## Static validation completed

- JSON Schema validation for all records
- stable and unique IDs
- legacy-ID preservation
- empty-gold consistency
- no-markdown gold policy
- continuation-semicolon policy
- declared substring constraints
- UTF-8 and UTF-16 cursor offsets
- required-object presence in the frozen catalog text
- deterministic record and package hashes

## Runtime gates still required before scientific freeze

Static packaging is not proof that every SQL fragment parses, binds, compiles, or executes. The Lab 04 implementation must run the pending gates listed in `validation/runtime-validation.template.json`, including pinned Microsoft ScriptDom parsing, permission-filtered catalog binding, SQL fixture compilation, result-shape and sandbox-execution oracles, capability routing, leakage analysis, and independent reconstruction.
