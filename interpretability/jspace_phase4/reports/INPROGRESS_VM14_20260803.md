# IN PROGRESS — Phase 4.4 decision block, VM14

Updated: 2026-08-03 03:07 UTC. Phase 4 remains development-only and is not frozen.

## Recoverable boundary

- Source parent: `901fb4fc7578a913088c7947a2e6240f7fc45aeb` from clean `interp_jspace_part2`.
- Working branch: `interp_jspace_phase4_4`; remote branch exists.
- Terminal-B pre-commitment: pushed at `6f23a29896e61cc367ed884a2d840f0e08857f40`, before functional execution.
- M0 tests: 279 passed under the frozen package path.
- M0 A1000 tensor audit: pass; lens `6e48c773...f6bd6`, checkpoint `fd5a4ae...bf20`, header `b0cf4c8d...6d2a`, 63 exact finite layers.
- M0 Qwen snapshot/runtime: pass; exact revision `6a9e13bd...`, 23 files, 48 fused bindings.
- External published lens: exact hash `1718c8c...11e1`.
- M0 fresh-VM durability: 230/232 verified, with exactly two known historical deficits and zero unexpected failure.
- Fresh local registered-output backups now cover the A1000 fit (3 outputs; manifest `cdaae012...397c`) and A500--A1000 structural event (6 outputs; manifest `5bff1706...7e09`).
- The first functional attempt hard-stopped before any A500 row or evidence event because `topk(32)[:10]` is not an exact replay of `topk(10)` at a tied boundary. The unchanged parent selected an equally scoring ID; an exact-size diagnostic replay matched it.
- The outcome-blind repair retains hard parent ID/score equality and adds tie-boundary regressions. Focused tests pass 27/27; the full Phase 4 suite passes 282/282.
- The unregistered 57 MB partial state is preserved intact on Drive and locally under the paths and hashes in `PHASE4_PART4_FUNCTIONAL_INSTRUMENTATION_INCIDENT.md`. It will not be reused across commits.
- No functional event exists; the queue lock is free and the next attempt starts from a fresh canonical output directory.

## Exact next command

From `/content/labs_phase4_4`, run the sealed queue through its mechanical canonical decision:

```bash
JSPACE4_EXPECTED_BRANCH=interp_jspace_phase4_4 \
JSPACE4_STOP_AFTER=canonical \
HF_HUB_CACHE=/content/hf_local \
bash interpretability/jspace_phase4/run_qwen_a1000_postfit_queue.sh
```

Run this only from a clean, pushed repair commit. The wrapper will reverify the completed stages and restart the functional gate from scratch; it must not migrate the withdrawn partial state. Its Drive heartbeat is the active watchdog during model work. Every new scientific stage must register, back up, commit, and push before the wrapper continues.

## Hard boundaries

- Queue order is functional, selection margin, prompt-323 influence, canonical Q-L decision.
- M2 side admission follows the registered canonical event.
- P4-P1 stays estimation-only; P4-P3 stays blocked at 16/20.
- P4-P2 is the sole conditional primary and cannot run without an actual independent producer review.
- Do not self-sign review or PI fields, do not open confirmatory/replication outcomes, and do not fit A2000.
