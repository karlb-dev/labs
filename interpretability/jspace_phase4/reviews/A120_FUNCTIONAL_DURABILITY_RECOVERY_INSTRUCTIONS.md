# A120 functional-event durability recovery instructions

**ENGINEERING / GOVERNANCE REVIEW PACKET — NO SCIENTIFIC RESULT — REVIEW AND
PI DECISION PENDING FOR ANY REGISTRY CORRECTION**

The live event
`p4-qwen-multilens-functional-gate-a120-a250-published-dev-v1` has two absent
historical files. Every other registered output verifies exactly.

```text
state.json
expected 361bda08e9ffbe1d333fd3cfaf3c7b9545e6a3504246a16dd8b0c07ad26f45e8

capacity_reconstructions_a120.pt
expected 6b0399df2c57158e7fdad24274e50f8c1058021d233412afdcc5177f6c651b6f
```

These instructions distinguish exact byte restoration from append-only
registry correction. Never invent either file, weaken the registered hashes,
or treat a near match as recovery.

## 1. Exact A120 capacity restoration

The fixed capacity cache, A120 lens, original YAML, dictionary and pursuit
modules, and the ASTs of all capacity functions are pinned in
`p4_qwen_historical_capacity_recovery.yaml`. The capacity code is unchanged
from source commit `30b121d`. Independent evidence for determinism is already
present: the A250 and published reconstruction artifacts generated in the
A120--A250 and A250--A500 runs are byte-identical for each common lens.

After A1000 releases the GPU, from a clean commit run:

```bash
python -m jspace_phase4.experiments.p4_qwen_historical_capacity_recovery \
  --config interpretability/jspace_phase4/configs/p4_qwen_historical_capacity_recovery.yaml \
  --preflight

python -m jspace_phase4.experiments.p4_qwen_historical_capacity_recovery \
  --config interpretability/jspace_phase4/configs/p4_qwen_historical_capacity_recovery.yaml \
  --recover
```

The utility performs no model forward pass. It loads only the pinned LM head
and final norm, reconstructs the three frozen capacity layers, writes a local
candidate using the historical archive-name convention, and moves it to the
registered path only if the entire SHA-256 is exactly `6b0399df...51b6f`.
A mismatch is quarantined under `/content/sl4_work`, not registered or copied
to Drive as evidence.

## 2. `state.json` must not be synthesized

The state carried wall-clock `elapsed_seconds` and peak-allocation fields as
well as raw lens cells. Those timing bytes cannot be honestly inferred from
the preserved Parquets, result, caches, and reconstruction tensors. Search
Drive revisions, backups, and prior VM storage for the exact hash. If no exact
copy exists, do not rerun merely to manufacture a different state file.

The 2026-08-02 exact-copy search is recorded in
`A120_STATE_EXACT_COPY_SEARCH_20260802.md`. It exhausts the mounted tree, live
Drive/trash listing, Drive revision surface, and preserved pre-incident
DriveFS metadata available on this VM without finding a target cloud ID or
exact bytes. A genuinely new backup may still be checked against the
registered SHA-256; the negative audit does not authorize synthesis or choose
an append-only resolution.

An independent reviewer and the PI must choose one append-only resolution:

1. keep the original event live and accept that whole-registry durability
   remains red; or
2. create a new durability-corrected archival event which pins every verified
   preserved scientific output plus a correction record, explicitly omits the
   unrecoverable implementation state, and then supersede the old event.

Option 2 must preserve the original event forever in registry history, state
that no scientific estimate or claim changed, demonstrate that the result,
input manifest, raw outcome tables, fixed activation caches, selection pairs,
capacity reconstructions, and figures all verify, and explain why `state.json`
is not required to recompute or audit the registered development conclusion.
The implementation agent may prepare tooling but may not self-approve this
choice or append the correction without the external decision.

## 3. Required review record

Return a JSON file containing at least:

```json
{
  "schema_version": 1,
  "review_scope": "p4-a120-functional-durability-recovery-v1",
  "reviewer_identity": "<independent reviewer identity>",
  "reviewer_is_independent": true,
  "reviewed_commit": "<full commit>",
  "reviewed_utc": "<UTC timestamp>",
  "findings": {
    "capacity_exact_hash_gate_approved": true,
    "capacity_near_match_cannot_be_restored": true,
    "state_exact_copy_search_adequate": true,
    "state_synthesis_forbidden": true,
    "append_only_resolution_selected": true,
    "scientific_claim_unchanged": true
  },
  "selected_state_resolution": "KEEP_RED_OR_SUPERSEDE_WITH_ARCHIVAL_EVENT",
  "verdict": "APPROVE_ENGINEERING_RECOVERY"
}
```

Reviewer identity, independence, commit binding, UTC time, findings, selected
resolution, and verdict must be real. Placeholders, self-signatures, a dirty
reviewed tree, or a scientific reinterpretation invalidate the review.
