# Phase 4 side-track import-bundle contract

Gemma and OLMo jobs own isolated packages, Drive roots, branches, and
append-only registries. They may never append `gm-*` or `ol-*` events to the
Phase 4 registry. Phase 4 imports only a hash-pinned bundle after the side
branch is pulled and merged with ancestry preserved.

## Envelope

Each bundle is a JSON envelope with `schema_version`, `payload`, and the
canonical `payload_sha256`. The payload contains:

```json
{
 "schema_version": 1,
 "bundle_id": "ol-phase4-bank-w-capability-v1",
 "source_study": "jspace-olmo-lineage",
 "source_branch": "interp_jspace_olmo_lineage",
 "source_commit": "<40-hex merged side commit>",
 "evidence_id_prefix": "ol-",
 "source_registry": {
  "path": "<side reports/evidence_events.jsonl>",
  "sha256": "<64-hex>"
 },
 "selected_events": [
  {
   "evidence_id": "ol-...",
   "role": "bank-w-baseline-capability",
   "contains_untouched_intervention_outcome": false
  }
 ],
 "target": {
  "study_id": "jspace-phase4",
  "import_evidence_id": "p4-import-olmo-bank-w-capability-v1"
 },
 "governance": {
  "development_or_methods_only": true,
  "contains_confirmatory_intervention_outcomes": false,
  "contains_replication_intervention_outcomes": false,
  "mainline_registry_was_not_written_by_side_track": true
 }
}
```

Gemma substitutes source study `jspace-gemma-transport` and prefix `gm-`.

## Mechanical validation

The Phase 4 validator independently checks the bundle envelope, exact source
registry hash and study ID, live status of every selected event, namespace,
development/methods tier, reachable event and bundle commits, ancestry,
every output's bytes and SHA-256, governance assertions, and absence of a
target-ID collision. Superseded or withdrawn source evidence is rejected.

Validation is read-only:

```bash
python -m jspace_phase4.import_bundle \
  --bundle <bundle.json> \
  --output <local-validation.json>
```

After the side branch and bundle are merged and the repository is clean, a
fresh validation must exactly equal the saved validation before registration:

```bash
python -m jspace_phase4.experiments.p4_import_side_bundle \
  --bundle <bundle.json> \
  --validation <local-validation.json>
```

The resulting mainline event uses a `p4-import-*` ID and tier
`side-development-import`; it hashes the bundle, validation, and every source
output. It does not copy side events into the main registry or relabel them as
Phase 4 confirmatory evidence.
