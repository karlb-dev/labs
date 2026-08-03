# IN PROGRESS — Phase 4.4 decision block, VM14

Updated: 2026-08-03 02:37 UTC. Phase 4 remains development-only and is not frozen.

## Recoverable boundary

- Source parent: `901fb4fc7578a913088c7947a2e6240f7fc45aeb` from clean `interp_jspace_part2`.
- Working branch: `interp_jspace_phase4_4`; remote branch exists.
- Terminal-B pre-commitment: pushed at `6f23a29896e61cc367ed884a2d840f0e08857f40`, before functional execution.
- M0 tests: 279 passed under the frozen package path.
- M0 A1000 tensor audit: pass; lens `6e48c773...f6bd6`, checkpoint `fd5a4ae...bf20`, header `b0cf4c8d...6d2a`, 63 exact finite layers.
- M0 Qwen snapshot/runtime: pass; exact revision `6a9e13bd...`, 23 files, 48 fused bindings.
- External published lens: exact hash `1718c8c...11e1`.
- M0 fresh-VM durability: 230/232 verified, with exactly two known historical deficits and zero unexpected failure.
- No model writer or queue lock holder existed at M0; no untouched intervention outcome exists.

## Exact next command

From `/content/labs_phase4_4`, first recreate and verify local registered-output backups without opening the functional gate:

```bash
JSPACE4_EXPECTED_BRANCH=interp_jspace_phase4_4 \
JSPACE4_STOP_AFTER=structural \
HF_HUB_CACHE=/content/hf_local \
bash interpretability/jspace_phase4/run_qwen_a1000_postfit_queue.sh
```

After that command exits at the already registered structural boundary, verify both backup manifests, bank this handoff, and rerun the same wrapper with `JSPACE4_STOP_AFTER=canonical`. The wrapper's Drive heartbeat is the active watchdog during model work.

## Hard boundaries

- Queue order is functional, selection margin, prompt-323 influence, canonical Q-L decision.
- M2 side admission follows the registered canonical event.
- P4-P1 stays estimation-only; P4-P3 stays blocked at 16/20.
- P4-P2 is the sole conditional primary and cannot run without an actual independent producer review.
- Do not self-sign review or PI fields, do not open confirmatory/replication outcomes, and do not fit A2000.
