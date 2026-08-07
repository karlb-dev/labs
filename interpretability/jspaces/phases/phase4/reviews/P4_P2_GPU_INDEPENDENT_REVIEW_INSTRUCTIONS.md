# P4-P2 GPU independent-review instructions

This packet is for a reviewer who is independent of the implementation. It
does not authorize the implementation agent to sign, fabricate, or register a
review on the reviewer's behalf. Review is development-only and must not open
any intervention outcome, untouched family, or confirmatory/replication data.

## Scope

Review the final bound versions of:

- `configs/p4_qwen_mode_variance_pilot_gpu_dev.yaml`;
- `jspace_phase4/experiments/p4_qwen_mode_variance_gpu.py`;
- `jspace_phase4/mode_intervention.py`;
- `tests/test_qwen_mode_variance_gpu.py`;
- `tests/test_mode_intervention.py`.

The config presently contains two explicit A1000 placeholders. Review may
start now, but the registered approval must hash the later config after those
placeholders are replaced. The producer verifies every reviewed file hash at
execution time.

The reviewer must independently inspect phase ownership, clean/intervened KV
cache separation, clean per-position protection alignment, exact instantaneous
rank/energy control construction, parser and exact-alias grading, immutable
resume semantics, and the development-only data boundary. Run at minimum:

```bash
prep_py="$PWD/interpretability/jspaces/phases/phase4:$PWD/interpretability/jspaces/phases/phase3:$PWD/interpretability/jspaces/phases/phase2"
PYTHONPATH="$prep_py${PYTHONPATH:+:$PYTHONPATH}" \
  pytest -q \
  interpretability/jspaces/phases/phase4/tests/test_mode_intervention.py \
  interpretability/jspaces/phases/phase4/tests/test_qwen_mode_variance_pilot.py \
  interpretability/jspaces/phases/phase4/tests/test_qwen_mode_variance_gpu.py
```

## Required payload

If and only if the review passes, author
`P4_P2_GPU_INDEPENDENT_REVIEW.json` as a standard Phase 4 result envelope with
evidence ID `p4-review-qwen-mode-variance-gpu-v1`, tier `methods`, and this
payload schema. Compute every SHA-256 from the final reviewed files; do not
copy the provisional values below after A1000 binding changes the config.

```json
{
  "schema_version": 1,
  "evidence_id": "p4-review-qwen-mode-variance-gpu-v1",
  "reviewer_identity": "<independent reviewer identity>",
  "review_completed_utc": "<YYYY-MM-DDTHH:MM:SSZ>",
  "reviewed_config_sha256": "<final config SHA-256>",
  "reviewed_producer_sha256": "<producer SHA-256>",
  "reviewed_mode_intervention_sha256": "<hook module SHA-256>",
  "reviewed_gpu_test_sha256": "<GPU producer test SHA-256>",
  "reviewed_mode_test_sha256": "<hook test SHA-256>",
  "phase_ownership_approved": true,
  "cache_isolation_approved": true,
  "per_position_protection_alignment_approved": true,
  "exact_profile_control_approved": true,
  "parser_and_grading_approved": true,
  "immutable_resume_contract_approved": true,
  "development_only_boundary_approved": true,
  "intervention_outcome_opened": false,
  "untouched_family_opened": false,
  "confirmatory_or_replication_outcome_opened": false,
  "verdict": "APPROVE_DEVELOPMENT_PILOT",
  "review_notes": "<concise independent findings>"
}
```

Wrap the payload with `write_result4`, then register that single output through
the append-only Phase 4 registry. The producer will reject an unregistered
file, a dirty-tree provenance block, a missing identity/UTC stamp, any false
required finding, any file-hash drift, or a verdict other than
`APPROVE_DEVELOPMENT_PILOT`.

An adverse or conditional review must not use the approval verdict or be
registered under the approval evidence ID. Return findings to the mainline
for a superseding implementation and fresh review instead.
