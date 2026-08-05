# CANONICAL_SCHEMA.md — linked domain tables (P0/P3, frozen)

Do not force every study into one wide table; use linked domain tables.
**`estimator_version` and `primary_tier` (with `source_evidence_id`) are
mandatory keys in every table** so incompatible estimands cannot be
averaged silently (P3 gate). Parquet outputs live under
`interpretability/jspace_paper/analysis/data/`.

## T1 `model_checkpoint_lens_manifest.parquet`

```text
model_family, checkpoint_id, revision, lineage_parent_claim, architecture,
n_layers, d_model, tokenizer_hash, chat_template_hash, lens_id, lens_hash,
fit_corpus_hash, fit_n, fit_draw, source_layers, target_map, estimator,
source_tier
```

## T2 `causal_item_outcomes.parquet`

```text
study, phase, model, checkpoint, lens, item_id, canonical_family,
template_id, partition, capability_status, condition, protection_geometry,
phase_of_intervention, requested_rank, effective_rank, removed_energy,
baseline_score, intervention_score, paired_delta, accuracy, answer_tokens,
bridge_id, source_evidence_id
```

Rule: never join item rows across studies solely by text when a frozen
item ID exists.

## T3 `capacity_geometry.parquet`

Separate metrics, never conflated: occupancy crossing; centered excess
variance; raw reconstruction excess; operator similarity; mapped-token
row similarity; selected-ID overlap; projector overlap; principal angles;
identity fraction; readout metrics.

```text
study, model, checkpoint, lens, layer_or_stratum, metric_class, metric,
value, estimator_version, paper_comparable, source_evidence_id,
primary_tier
```

## T4 `transport_validation.parquet`

```text
model, checkpoint, prompt, layer, position, direction_family, epsilon,
injection_precision, backend_pair, backend_relative_error,
backend_ceiling, delivery_fidelity, snr, measurement_eligible,
decision_eligible, tangent_cosine, relative_error, central_relative_error,
gain, pass, route, source_evidence_id
```

No row may say "linear"/"nonlinear" without map, prompt, layer, dose, and
gate specified.

## T5 `qwen_lens_stability.parquet`

One row per comparison, metric, layer/stratum, endpoint:

```text
lens_a, lens_b, fit_n_a, fit_n_b, metric_class, metric, value, threshold,
pass, assay_band, source_evidence_id
```

Nested same-corpus comparisons and the partially specified published-lens
comparator are distinct `metric_class` families, never mixed.

## T6 `capability_gating.parquet` and `missingness_register.parquet`

```text
model, checkpoint, bank, family, item, capability_direct,
capability_composed, capability_joint, cohort_requirement, cohort_status,
missing_reason, gate_evidence_id
```

`missing_reason` uses the frozen state vocabulary from
`EVIDENCE_TIER_RULES.md` rule 3.

## T7 `master_evidence_events.parquet`

One row per registry event across all six sources:

```text
registry, evidence_id, event_type, created_utc, tier, status,
superseded_by, supersedes, imports_from, title_or_kind, payload_sha256,
source_line_no
```

## T8 `master_claim_ledger.parquet` + `master_claim_evidence_edges.parquet`

Claim rows:

```text
claim_id, claim_text_short, claim_text_maximum_licensed, scope_models,
scope_tasks, scope_layers, scope_doses, scope_estimator, primary_tier,
evidence_ids, replication_ids, falsifiers, status, forbidden_upgrades,
paper_route
```

Edge types (the paper's spine — a claim with no valid `supports` edge
cannot appear as a result sentence):

```text
supports, narrows, falsifies, supersedes, replicates, bounds, blocks,
requires, imports, cannot_adjudicate
```

## Figure ledger schema (P7)

```text
figure_id, paper_route, scientific_question, source_evidence_ids,
source_table, analysis_script, primary_tier, retrospective_flag,
caption_claims, forbidden_caption_claims, reconstruction_status
```

Reconstruction status vocabulary (P4 §7.2):

```text
byte_identical, numerically_identical_render_diff,
numerically_within_frozen_tolerance, failed,
not_reconstructable_from_released_data
```
