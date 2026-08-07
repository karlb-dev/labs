# J-space Phase 4

Phase 4 asks what computation occupies the verbalizable channel, how
post-training reroutes it, when controlled load engages it, and when the
J-lens is a valid transport instrument.

Phase 2 and Phase 3 are closed inputs. Their artifacts enter this package
only through immutable import events. The Phase 4 preregistration remains a
candidate: **no Phase 4 confirmatory or replication model cell is authorized
until PI sign-off and a freeze tag exist.**

## GPU invariant

Every model-scale entrypoint must call
`jspace_phase4.gpu.require_cuda_gpu()` in the same process before loading
weights and must call `assert_model_on_cuda(model)` after load. A sandboxed
CUDA failure is a hard stop and a request to relaunch the exact command with
host GPU access; it is never permission to fall back to CPU.

CPU is reserved for conformance tests, hashing, state/manifest handling,
statistics, plotting, and document compilation.

## Quick start

```bash
export JSPACE4_RUN_ROOT=/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731
bash interpretability/jspaces/phases/phase4/repro.sh
jspace-phase4 registry-list
jspace-phase4 verify
```

The run root, model cache, Phase 3 imports, and repository files are resolved
through logical URIs. Scientific modules must not embed machine paths.

## Phase 4.3 closeout and Phase 4.4 resume block

The registered post-A500 decision was Branch B. Its Qwen Draw A continuation
resumed from the durable n=554 VM handoff and completed at n=1000 under the
unchanged frozen wrapper. The final registered lens SHA-256 is
`6e48c7731501d0fc6030f1d60eff6f19b211756f40ef0cd6e499e414f08f6bd6`;
the final checkpoint SHA-256 is
`fd5a4ae614eef46002cc987a038d9a391016b7fbc91a754eed2adff83f6abf20`.
n=554 remains the immutable restart-provenance boundary:

```text
checkpoint SHA-256: bf992067d690123109198c182a21169379e5752d89e73e96514fab7127fba74d
checkpoint bytes:   6,606,047,399
next chunk:         554:557
```

n=554 is a restart boundary, not a registered evidence milestone. The final
n=1000 event and A500--A1000 structural successor are registered, backed up,
and pushed. Read both handoffs before Phase 4.4:

```bash
export JSPACE4_RUN_ROOT=/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731
export JSPACE4_LOCAL_WORK=/content/sl4_work
export HF_HUB_CACHE=/content/hf_local
cat /content/resume-phase-4-2.md
cat /content/drive/MyDrive/interpret/inprogress.md
```

The n=500 fit and four-stage post-fit queue are complete and registered. The
exact A500 hash is bound in the two successor YAMLs, and the structural,
functional, official mode-v2, and Qwen Bank-W capability events are pushed.
Do not rerun that queue. Apply the registered A/B/C branch without changing
thresholds:

```bash
bash interpretability/jspaces/phases/phase4/run_qwen_frozen_branch_followup.sh
```

The router and n=1000 fit must not be rerun. Their exact registered outputs are
immutable. The lower-level producer below remains only for recovery
diagnostics and contract verification.

The lower-level producer remains available for exact recovery diagnostics:

```bash
python -m jspace_phase4.experiments.p4_qwen_nested_lens_fit \
  --config interpretability/jspaces/phases/phase4/configs/p4_qwen_nested_lens_fit_dev.yaml \
  --draw draw_a --stop-at 1000
```

The handoff-time whole-registry audit accounts for 218/220 live outputs with
no hash mismatch. Two files from the older A120--A250 functional-gate event
are absent: `state.json` (expected SHA-256 `361bda08...f45e8`) and
`capacity_reconstructions_a120.pt` (expected `6b0399df...51b6f`). The missing
published reconstruction was restored from an exact hash-verified backup.
See the handoffs for full paths and hashes; do not fabricate the remaining
bytes or edit the append-only registry. Candidate 0.11 adds a hash-gated A120
capacity recovery which may run only after A1000 releases the GPU and only
installs a bit-exact match. `state.json` remains an external-review/PI
governance decision if no exact backup can be found. The mounted tree, live
Drive/trash and revision surfaces, and preserved pre-incident DriveFS metadata
were searched without finding a target cloud ID or exact bytes; see
`reviews/A120_STATE_EXACT_COPY_SEARCH_20260802.md`.

The n=250 convergence, retained prompt-112 influence, functional Branch-B
gate, A500 fit and successor gates, passing mode-v2 baseline, Qwen Bank-W
capability, mode-parser v2 contract, Bank B feasibility, Bank-W protocol, and
conditional P4-P2 variance-pilot protocol are registered. Do not rerun or
overwrite them; registry verification checks their immutable hashes.

Candidate 0.11 keeps Bank B/P4-P1 estimation-only while preparing the mandated
single consumed-development orthogonal feasibility shot, fixes the pre-pilot
0.20-point P4-P2 SESOI, and now includes the executable GPU producer plus its
mean-masked exact/Monte-Carlo power ruler. The OLMo capability service result
fails at 16/20 common families, so P4-P3 is blocked. The A500--A1000 successor
configs, selection-margin contract, prompt-323 influence contract, and
conditional Q-L2 estimand were committed before A1000 existed. The exact
registered A1000 hash is now bound in the three authorized slots without
changing thresholds.

The structural event passes at conservative task q50/q05
0.998702/0.998122. Phase 4.4 has also registered and backed up the functional
and selection-margin stages. They emit provisional Q-L4: A500--A1000
selected-ID Jaccard, selected-projector overlap, and bridge rescue fail their
frozen gates. The required prompt-323 influence stage then hard-stopped before
writing a contribution because repeated current-runtime Jacobian norms fail
the frozen fit-log control; repeated prompt-112 controls prove a broader
backward-runtime mismatch. No influence or canonical event exists.

Do **not** rerun `run_qwen_a1000_postfit_queue.sh` blindly. Continuation
requires independent review and either a historically content-pinned runtime
or a prospective runtime-contract amendment before a fresh influence attempt.
The 0.5 tolerance remains frozen, neither null state may be resumed, and Q-L4
may not be registered directly. See
`reports/PHASE4_PART4_PROMPT323_RUNTIME_BLOCK.md` and its machine-readable JSON.

Gemma and OLMo completed on isolated branches and registries, and their full
ancestry is merged without copying native `gm-*` or `ol-*` events into Phase
4. The early OLMo service bundle, terminal Gemma methods blocker, and terminal
OLMo methods release are strictly normalized and queued behind A1000 under
`protocol/SIDE_TRACK_IMPORT_BUNDLE_CONTRACT.md`. Their exact boundaries are
recorded in `manifests/parallel_import_inventory.md`; a validated bundle is
not registered Phase 4 evidence until its single `p4-import-*` event lands on
mainline. `run_phase4_post_a1000_import_queue.sh` enforces the prescribed
admission and joint-replay order after the canonical decision.
Whole-registry release requires the two-pass procedure in
`protocol/DRIVEFS_DURABILITY_PLAN.md`; known deficits remain failures.
After model writers stop and the registry is final, generate the review
inventory with `python -m jspace_phase4.pre_freeze_inventory`. It hashes every
live output, checks reachable commits and namespace/path policy, and writes the
required JSON/Markdown manifests; a known deficit remains a failed gate.

## Current development synthesis

- Living Markdown: `reports/PHASE4_DEVELOPMENT_REPORT.md`
- Candidate freeze-gate ledger:
  `preregistration/FREEZE_GATE_LEDGER_PHASE4.md`
- Methods decision record: `paper/PHASE4_METHODS_DECISION_RECORD.md`
- Falsifiable conclusion skeleton: `paper/PAPER_CONCLUSION_SKELETON.md`
- Compiled handout through the pre-canonical boundary (regenerate only after the
  canonical decision): `reports/handout/jspace_phase4_development.{tex,pdf}`
- Phase 4.3 closeout and exact Phase 4.4 handoff:
  `reports/phase4_part3_summary.md`
- Open A1000 raw-diagnostic archival review:
  `reviews/QWEN_A1000_RAW_DIAGNOSTIC_ARCHIVE_REVIEW_20260802.md`
- Current durable restart snapshot:
  `reports/INPROGRESS_VM14_20260803.md`
- Prompt-323 runtime-identity execution block:
  `reports/PHASE4_PART4_PROMPT323_RUNTIME_BLOCK.md`
- Historical VM12 restart ledger:
  `reports/INPROGRESS_VM12_20260801.md`
- Compact restart handoff:
  `reports/RESUME_PHASE4_2.md`
- Governing development block:
  `reviews/jspace_lab_nextsteps_4_4.md` plus
  `reviews/jspace_lab_nextsteps_4_4_addendum.md`

These documents now cover the base, 3.0 Think, and sibling 3.1
Think/Instruct capability and intervention points, seed-paired
own/common-frame audits, and the registered four-checkpoint trajectory
synthesis. Phase 4.2 adds the Qwen same-corpus convergence and functional
invariance gates, CPU-first common-cohort closure, Bank B/W authoring and
power, Bank W capability rules, official-mode gates, and P4-P2 design/pilot
methods. Phase 4.3 closes the A1000 fit and structural boundary and merges the
side-track ancestry. Phase 4.4 has completed the registered functional and
margin stages but is blocked at the required prompt-influence runtime identity;
the canonical decision, side-track admission, and later freeze blockers remain
closed. These remain development summaries, not frozen claims.

## Tiers

- `phase2-confirmatory-import`
- `phase3-confirmatory-import`
- `phase3-replication-import`
- `side-development-import`
- `phase4-development`
- `phase4-confirmatory`
- `phase4-replication`
- `methods`

Native Phase 2/3 evidence creation is rejected. Imported events name and
hash their source registry, source commit/tag, evidence ID, and outputs.
