# Gemma transport Phase 4 import bundle

This is a methods-only side-track export. The canonical machine-readable
envelope is `IMPORT_BUNDLE_PHASE4.json` in the isolated Drive release
directory.

Importable conclusion: the Gemma transport study stopped at a precommitted
actual-model cross-backend gate. The selected replay is exact, but the full
eight-slot tangent relative error is 0.002458 against a 1e-5 ceiling. No
mechanism, late-band, workspace, confirmatory, or replication conclusion is
importable.

Importers must verify:

1. the envelope payload SHA-256;
2. source branch and producer commit;
3. the exact registry prefix byte count and SHA-256;
4. all source evidence and output hashes in the inventory;
5. `backend_parity_pass: false` and the sole failed
   `backend_tangent_all_slots` criterion;
6. the claim ledger and transport-gate protocol hashes.

The bundle does not authorize Phase 4 intervention execution, independent
review, PI sign-off, or a conclusion that information or a workspace is
absent. Any future repair uses a new evidence ID and preserves the failed
artifact.
