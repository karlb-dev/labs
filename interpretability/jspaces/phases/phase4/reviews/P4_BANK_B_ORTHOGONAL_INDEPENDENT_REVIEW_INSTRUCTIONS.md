# Bank-B orthogonal-rescue independent-review instructions

This packet is for a reviewer independent of the implementation. It does not
authorize the implementation agent to sign, fabricate, or register a review
on the reviewer's behalf. Review is methods/development only and must not open
any Bank-B candidate, confirmatory, or replication outcome.

## Scope

Review the final A1000-bound versions of:

- `configs/p4_bank_b_orthogonal_feasibility_dev.yaml`;
- `jspace_phase4/experiments/p4_bank_b_orthogonal_feasibility.py`;
- `jspace_phase4/orthogonal_bridge.py`;
- `jspace_phase4/bank_b_orthogonal_analysis.py`;
- `tests/test_orthogonal_bridge.py`;
- `tests/test_bank_b_orthogonal_analysis.py`;
- `tests/test_bank_b_orthogonal_feasibility.py`.

The config currently contains two explicit A1000 placeholders. Review may
start now, but approval must hash the final config after binding. The producer
rechecks every reviewed hash at outcome execution.

Independently verify that the source reader requests only `fact_id` and
`canonical_family` from the registered consumed Phase-3 Parquet; all
`bank-b:*` facts and every confirmatory/replication outcome are refused; the
answer span is the union of all original and counterfactual alias pieces;
unrelated matching uses geometry only; the clean protection rows align by
position; the true-bridge lesion is sign-independent; selected/effective rank
matches across all six arms; each unit direction receives exactly the removed
norm; hooks are prompt-only and absent during decode; state parts are atomic
and hash-bound; and the fixed 0.25-nat SESOI cannot be increased from the
observed mean.

Run at minimum:

```bash
prep_py="$PWD/interpretability/jspaces/phases/phase4:$PWD/interpretability/jspaces/phases/phase3:$PWD/interpretability/jspaces/phases/phase2"
PYTHONPATH="$prep_py${PYTHONPATH:+:$PYTHONPATH}" \
  pytest -q \
  interpretability/jspaces/phases/phase4/tests/test_orthogonal_bridge.py \
  interpretability/jspaces/phases/phase4/tests/test_bank_b_orthogonal_analysis.py \
  interpretability/jspaces/phases/phase4/tests/test_bank_b_orthogonal_feasibility.py

PYTHONPATH="$prep_py${PYTHONPATH:+:$PYTHONPATH}" \
  python -m jspace_phase4.experiments.p4_bank_b_orthogonal_feasibility \
  --config interpretability/jspaces/phases/phase4/configs/p4_bank_b_orthogonal_feasibility_dev.yaml \
  --preflight
```

If the outcome-blind geometry event already exists, verify all its registered
hashes and confirm `causal_language_model_forward_calls == 0`,
`selection_used_outcomes == false`, and `bank_b_outcomes_opened == false`.
Do not run `--run` as part of review.

## Required payload

If and only if review passes, author
`P4_BANK_B_ORTHOGONAL_INDEPENDENT_REVIEW.json` as a standard Phase-4 result
envelope with evidence ID `p4-review-bank-b-orthogonal-producer-v1`, tier
`methods`, and this payload schema. Compute hashes from the final reviewed
files rather than copying provisional values.

```json
{
  "schema_version": 1,
  "evidence_id": "p4-review-bank-b-orthogonal-producer-v1",
  "reviewer_identity": "<independent reviewer identity>",
  "reviewer_independent": true,
  "review_completed_utc": "<YYYY-MM-DDTHH:MM:SSZ>",
  "reviewed_config_sha256": "<final config SHA-256>",
  "reviewed_producer_sha256": "<producer SHA-256>",
  "reviewed_geometry_core_sha256": "<orthogonal_bridge.py SHA-256>",
  "reviewed_analysis_sha256": "<bank_b_orthogonal_analysis.py SHA-256>",
  "reviewed_core_test_sha256": "<test_orthogonal_bridge.py SHA-256>",
  "reviewed_analysis_test_sha256": "<test_bank_b_orthogonal_analysis.py SHA-256>",
  "reviewed_producer_test_sha256": "<test_bank_b_orthogonal_feasibility.py SHA-256>",
  "consumed_cohort_and_outcome_firewall_approved": true,
  "answer_span_and_geometry_approved": true,
  "unrelated_match_approved": true,
  "lesion_rank_and_dose_approved": true,
  "prompt_only_phase_approved": true,
  "scorer_and_generation_grader_approved": true,
  "immutable_resume_contract_approved": true,
  "variance_and_power_decision_rule_approved": true,
  "intervention_outcome_opened": false,
  "bank_b_outcome_opened": false,
  "confirmatory_or_replication_outcome_opened": false,
  "verdict": "APPROVE_CONSUMED_DEVELOPMENT_SHOT",
  "review_notes": "<concise independent findings>"
}
```

Wrap the payload with `write_result4`, then register that single output in the
append-only Phase-4 registry. An adverse or conditional review must not use
the approval verdict or approval evidence ID; return findings for a
superseding implementation and fresh review.
